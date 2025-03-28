import os, logging, time, shutil, json, re, sys
import random
import numpy as np
import pandas as pd
from scipy.signal import convolve2d
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanAbsoluteError
#import tensorflow_addons as tfa
from astropy.io import fits
from network import network

class Callback(tf.keras.callbacks.Callback):
    def __init__(self, dataset, dataset_size, epoch, start_epoch, eval_percent,
                 period_save, save_freq, sample_generator, kwargs_validation,
                validation_images, optimizer, change_learning_rate, batch_size, scaling):
        """
    Custom Keras callback for handling various training and validation tasks.

    This callback includes periodic model saving, validation, training metric tracking, 
    learning rate adjustments, and logging.

    Attributes:
        dataset (tf.data.Dataset): The training dataset.
        dataset_size (int): Total number of samples in the dataset.
        kwargs_validation (dict): Arguments for validation data creation.
        eval_percent (float): Percentage of the dataset used for validation.
        period_save (float): Frequency (percentage) of training samples saved.
        save_freq (int): Frequency (in epochs) to save models.
        epoch (int): Total number of epochs for training.
        current_epoch (int): The current epoch during training.
        best_loss (float): Tracks the best validation loss observed so far.
        training_metrics (list): List of dictionaries storing metrics for each epoch.
        start_time (float): Time when training started.
        sample_generator (function): Generator function for creating training samples.
        validation_images (list): List of validation image file paths.
        optimizer (tf.keras.optimizers.Optimizer): The optimizer used for training.
        change_learning_rate (list): List of tuples specifying epoch and corresponding learning rates.
        """
        super().__init__()
        self.dataset = dataset
        self.dataset_size = dataset_size
        self.kwargs_validation = kwargs_validation
        self.eval_percent = eval_percent
        self.period_save = period_save
        self.save_freq = save_freq
        self.epoch = epoch
        self.current_epoch = start_epoch
        self.best_loss = np.inf
        self.training_metrics = []
        self.start_time = time.time()
        self.sample_generator = sample_generator
        self.validation_images = validation_images
        self.optimizer = optimizer
        self.change_learning_rate = change_learning_rate
        self.lr = 0
        self.batch_size = batch_size
        self.scaling = scaling

    def create_directory(self, path):
        """
        Creates a directory if it does not already exist.

        Args:
            path (str): The path of the directory to create.
        """
        ...
        try:
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
        except Exception as err:
            logging.warning(f"An error occurred while creating the directory {path}: {err}")

    def on_train_begin(self, logs=None):
        """
        Called at the beginning of training.

        Args:
            logs (dict, optional): Training logs. Defaults to None.
        """
        self.train_start = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        """
        Called at the beginning of each epoch. Adjusts learning rate if specified.

        Args:
            epoch (int): The current epoch index.
            logs (dict, optional): Training logs. Defaults to None.
        """
        self.epoch_start = time.time()
        for change_epoch, lr in self.change_learning_rate:
            if self.current_epoch >= change_epoch:
                if lr != self.lr:
                    self.optimizer.learning_rate.assign(lr)
                    self.lr = lr
                    logging.warning(f'epoch: {self.current_epoch} changing to lr: {lr}')

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each epoch. Performs periodic model saving, 
        validation, and metric tracking.

        Args:
            epoch (int): The current epoch index.
            logs (dict, optional): Training logs, including the training loss. Defaults to None.
        """
                
        self.current_epoch += 1
        train_loss = logs.get('loss', 0) if logs else 0
        #train_loss = train_loss / self.batch_size
        train_loss = train_loss
        epoch_time = time.time() - self.epoch_start 

        # Save training images periodically
        if self.current_epoch % max(int(self.save_freq), 1) == 0:
            result_path = f"./results/{self.current_epoch:04d}"
            self.create_directory(result_path)

            count = self.dataset_size * max(int(self.period_save), 0.1) // 100
            temp = 0
            indices = set()
            while temp < count:
                index = np.random.randint(0, sum(1 for _ in self.dataset))
                if index not in indices:
                    batch = self.dataset.take(index)
                    for x_trains, y_trains, filepaths in batch:
                        predictions = self.model.predict(x_trains)
                        for x_train, y_train, prediction, filepath in zip(x_trains, y_trains, predictions, filepaths):
                            filepath = np.array([fp.decode('utf-8') for fp in filepath.numpy()])  # Decode properly
                            org_img = noise_img = prd_img = None
                            if self.scaling == 'log_min_max' and len(filepath) == 7:
                                filepath, org_min, org_max, org_shift, noise_min, noise_max, noise_shift = filepath
                                org_min, org_max, org_shift = map(float, [org_min, org_max, org_shift])
                                noise_min, noise_max, noise_shift = map(float, [noise_min, noise_max, noise_shift])

                                sv_name = "/" + filepath.split('/')[-1][:-5]  # Extract filename
                                org_img = inverse_adaptive_log_transform_and_denormalize(y_train[:, :, 0], org_min, org_max, org_shift)
                                noise_img = inverse_adaptive_log_transform_and_denormalize(x_train[:, :, 0], noise_min, noise_max, noise_shift)
                                prd_img = inverse_adaptive_log_transform_and_denormalize(prediction[:, :, 0], noise_min, noise_max, noise_shift)

                            elif self.scaling == 'z_scale' and len(filepath) == 5:
                                filepath, org_mean, org_std, noise_mean, noise_std = filepath
                                org_mean, org_std = map(float, [org_mean, org_std])
                                noise_mean, noise_std = map(float, [noise_mean, noise_std])

                                sv_name = "/" + filepath.split('/')[-1][:-5]  # Extract filename
                                org_img = inverse_zscore_normalization(y_train[:, :, 0], org_mean, org_std)
                                noise_img = inverse_zscore_normalization(x_train[:, :, 0], noise_mean, noise_std)
                                prd_img = inverse_zscore_normalization(prediction[:, :, 0], noise_mean, noise_std)
                            elif self.scaling == 'min_max' and len(filepath) == 5:
                                filepath, org_min, org_max, noise_min, noise_max = filepath
                                org_min, org_max = map(float, [org_min, org_max])
                                noise_min, noise_max = map(float, [noise_min, noise_max])

                                sv_name = "/" + filepath.split('/')[-1][:-5]  # Extract filename
                                org_img = inverse_min_max_normalization(y_train[:, :, 0], org_min, org_max)
                                noise_img = inverse_min_max_normalization(x_train[:, :, 0], noise_min, noise_max)
                                prd_img = inverse_min_max_normalization(prediction[:, :, 0], noise_min, noise_max)
                            else:
                                filepath = filepath[0]
                                sv_name = "/" + filepath.split('/')[-1][:-5]
                                org_img = y_train[:, :, 0]
                                prd_img = prediction[:, :, 0]
                                noise_img = x_train[:, :, 0]
                            #if isinstance(org_img, np.ndarray) and isinstance(prd_img, np.ndarray) and isinstance(noise_img, np.ndarray):
                            save_fits(org_img, sv_name, result_path)
                            save_fits(prd_img, sv_name + "_output", result_path)
                            save_fits(noise_img, sv_name + "_noise", result_path)
                            temp += 1
                    indices.add(index)

        # PERIODIC VALIDATION
        self.validation_start = time.time()
        count = self.dataset_size * max(int(self.eval_percent), 1) // 100
        indices = np.random.randint(0, len(self.validation_images), 
                                     size=min(count, len(self.validation_images)))
        selected_images = np.array(self.validation_images)[indices]
        validation_data = create_tf_dataset(selected_images, self.sample_generator,
                                                 generator_kwargs=self.kwargs_validation, 
                                                 scaling=self.scaling)
        validation_loss = 0
        num_batches = 0
        for x_val, y_val, _ in validation_data:
            predictions = self.model(x_val, training=False)
            loss_value = self.model.compiled_loss(y_val, predictions)
            validation_loss += float(loss_value)
            num_batches += 1
        avg_validation_loss = validation_loss / num_batches

        # Track best validation loss
        if avg_validation_loss < self.best_loss:
            self.best_loss = avg_validation_loss
            checkpoint_path = './checkpoints'
            self.create_directory(checkpoint_path)

            try:
                self.model.save(f'{checkpoint_path}/best_model_{self.current_epoch:04d}.keras')
            except Exception as err:
                logging.warning(f"An error occurred while saving the best model: {err}")

        try:
            with open(f"./validation_loss.txt", "a") as file:
                file.write(f"Epoch {self.current_epoch}: Validation Loss = {avg_validation_loss:.4f}, "
                           f"Time: {(time.time() - self.validation_start):.2f}s\n")
        except Exception as err:
            logging.warning(f"An error occurred while saving the validation results: {err}")

        #PERIODIC SAVING OF METRICS
        current_metrics = {
        'epoch': self.current_epoch,
        'train_loss': train_loss,
        'validation_loss' : avg_validation_loss,
        'epoch_time': epoch_time,
        }
        self.training_metrics.append(current_metrics)
        csv_path = './training_history.csv'
        # Load existing CSV or create a new one
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df = pd.concat([df, pd.DataFrame([current_metrics])], ignore_index=True)
        else:
            df = pd.DataFrame([current_metrics])
        # Save updated dataframe back to the CSV file
        try:
            df.to_csv(csv_path, index=False)
        except Exception as err:
            logging.warning(f"An error occurred while saving the CSV file: {err}")

        # PERIODIC MODEL SAVE
        if self.current_epoch % max(int(self.save_freq), 1) == 0:
            checkpoint_path = './checkpoints'
            self.create_directory(checkpoint_path)

            try:
                self.model.save(f'{checkpoint_path}/model_{self.current_epoch:04d}.keras')
            except Exception as err:
                logging.warning(f"An error occurred while saving the periodic model: {err}")

    def on_train_end(self, logs=None):
        """
        Called at the end of training. Saves the final model and training metrics.

        Args:
            logs (dict, optional): Training logs. Defaults to None.
        """
  
        checkpoint_path = './checkpoints'
        self.create_directory(checkpoint_path)

        try:
            final_model_path = f'{checkpoint_path}/final_model.keras'
            self.model.save(final_model_path)
        except Exception as err:
            logging.warning(f"An error occurred while saving the final model: {err}")

        # Save the training metrics (loss)
        training_time = time.time() - self.start_time
        try:
            with open(f'./training_metrics.txt', 'w') as file:
                file.write(f"Training completed in {training_time:.2f} seconds\n")
                file.write(f"Epoch Metrics (Epoch, Loss, Time):\n")
                for epoch, metrics in enumerate(self.training_metrics):
                    file.write(f"Epoch: {epoch:.4f}" + "\t" + 
                               f"Train_loss: {metrics.get('train_loss', 0):.4f}" + "\t" + 
                               f"Val_loss: {metrics.get('validation_loss', 0):.4f}" + "\t" + 
                               f"Time: {metrics.get('epoch_time', 0):.2f}\n")
        except Exception as err:
            logging.warning(f"An error occurred while writing the training stats: {err}")
############DATA HANDLING ##################################
############################################################
###########################################################
def open_fits(filepath, ratio, type_of_image='SCI', low=60, high=10000):
    """
    USED WITH 'data_augment()' function for images where data is given in electrons

    Opens a .fits file, extracts the specified image data and exposure time, 
    and filters based on the adjusted exposure time.

    Args:
        filepath (str): The path to the .fits file.
        ratio (float): The adjustment factor for the exposure time.
        type_of_image (str, optional): The image type to extract (e.g., 'SCI'). 
                                       Defaults to 'SCI'.
        low (int, optional): The lower limit for the adjusted exposure time. 
                             Defaults to 60.
        high (int, optional): The upper limit for the adjusted exposure time. 
                              Defaults to 10000.

    Returns:
        tuple or None: Returns a tuple `(image_data, filepath, ex_time, ratio)` if 
                       the file meets the criteria, otherwise returns `None`.

    Raises:
        Warning: Logs a warning if an error occurs while reading the .fits file.
    """
    try:
        ratio = float(ratio)
        with fits.open(filepath) as f:
            out, ex_time = None, None
            
            for hdu in f:
                if isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU)) and isinstance(hdu.data, np.ndarray):
                    if out is None and type_of_image.lower() == str(hdu.name).lower():
                        out = hdu.data
                if ex_time is None and 'EXPTIME' in hdu.header:
                    ex_time = hdu.header['EXPTIME']
                if out is not None and ex_time is not None:
                    break
            
            if out is None or ex_time is None or ex_time * ratio < low or ex_time * ratio > high:
                return None
            return out, filepath, ex_time, ratio
    except Exception as err:
        logging.warning(f'Error occurred while reading the .fits file: {err}')
        return None

def open_fits_2(filepath):
    """
    USED in data_augment_2() and data_augment_3() functions with images where data is given electron/s
    Opens a .fits file and extracts the first available image data.

    Args:
        filepath (str): The path to the .fits file.

    Returns:
        numpy.ndarray or None: The image data from the first valid HDU (Header Data Unit) 
                               if available, otherwise `None`.

    Raises:
        Warning: Logs a warning if an error occurs while reading the .fits file.
    """
    try:
        with fits.open(filepath) as f:
            for hdu in f:
                if isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU)) and hdu.data is not None and isinstance(hdu.data, np.ndarray):
                    return hdu.data
    except Exception as err:
        logging.warning(f"Error reading .fits file {filepath}: {err}")
    return None

def save_fits(image, name, path):
    """
    Saves an image as a .fits file. At the moment, the .fits files contain no metadata

    Args:
        image (numpy.ndarray): The image data to be saved.
        name (str): The name of the output .fits file (without extension).
        path (str): The directory path where the file will be saved.

    Returns:
        bool: Returns `True` if the file is saved successfully, otherwise `None`.

    Raises:
        Warning: Logs a warning if an error occurs while saving the .fits file.
    """
    try:
        hdu = fits.PrimaryHDU(image)
        hdul = fits.HDUList([hdu])
        hdul.writeto(path + name + '.fits', overwrite=True)
        return True
    except Exception as err:
        logging.warning(f'an error occurred while saving the image: {err}')
        return
    
#############HELPER FUNCTIONS###################################################
###############################################################################################################
###############################################################################################################
def linear_function(x, a, b):
    """
    Computes a linear function of the form y = ax + b.

    Args:
        x (float or numpy.ndarray): The input value(s).
        a (float): The slope of the line.
        b (float): The y-intercept of the line.

    Returns:
        float or numpy.ndarray: The computed value(s) of the linear function.
    """
    return a * x + b

def power_law(x, t, a):
    """
    Computes a power-law function of the form y = a * x^t.

    Args:
        x (float or numpy.ndarray): The input value(s).
        t (float): The exponent of the power-law function.
        a (float): The scaling factor.

    Returns:
        float or numpy.ndarray: The computed value(s) of the power-law function.
    """
    return a * (x ** t)

def set_threshold(k, start=False):
    """
    Computes a threshold value based on the input `k` which serves either as 10 to the k power
    or as k times depending on the flag 'start'. If start is set to True, it returns the highest
    value of the relation (a-b)/(a+b) where a is k times b. Otherwise of the value where a is 10^k times b
    This can be used to filter out too low to too high noisy sigma values with respect to the original sigma

    Args:
        k (int or float): The input value used to compute the threshold.
        start (bool, optional): If `True`, computes the threshold using the formula 
                                (k-1)/(k+1). If `False`, uses the formula 
                                (10^k-1)/(10^k+1). Defaults to `False`.

    Returns:
        float: The computed threshold value.
    """
    if start:
        return (k-1)/(k+1)
    else:
        return (10**k-1)/(10**k+1)
    
def is_not_nan(val):
    """
    Checks whether a value is a valid numeric value (not NaN, infinite, or NA).

    Args:
        val: The value to check. Can be of any type.

    Returns:
        bool: `True` if the value is a finite numeric value and not NaN or NA, otherwise `False`.
    """
    return isinstance(val, (float, np.floating)) and not np.isnan(val) and not np.isinf(val) and val is not pd.NA

def evenly_spaced_numbers(start, stop, num):
    """
    Generates a list of evenly spaced integers between `start` and `stop`.
    This function used to create a set of integers to serve as the base for augmented variance
    values of the form b*10**k. It also ensures that the original base is not included

    Args:
        start (int): The starting value of the range.
        stop (int): The ending value of the range (inclusive).
        num (int): The number of evenly spaced integers to generate.

    Returns:
        list: A list of `num` evenly spaced integers between `start` and `stop`.
              Returns an empty list if `stop <= start`, `num <= 0`, or if the
              range between `start` and `stop` is insufficient to generate
              the required number of integers.

    Notes:
        - If `start` is not equal to 1, it is incremented by 1 before generating the range.
        - The function ensures that the number of integers does not exceed the available range.
    """
    if not stop > start:
        return []
    if num <= 0:
        return []
    if start != 1:
        start += 1
    available_integers = stop - start + 1
    if available_integers < num:
        num = int(available_integers)
    if num <= 0:
        return [] 
    return np.linspace(start, stop, num).astype(int).tolist()

def find_columns(columns, suffixes, n):
    """
    Finds and filters column names based on suffixes and a numeric proximity to `n`.
    The initial idea was to use this function to filter out noisy images based on the calculated 
    statistics of the highest (100-n)% values of the original images. However, this seemed to filter
    almost all possible images so it is no longer used.

    Args:
        columns (list of str): A list of column names to search through.
        suffixes (list of str): A list of suffixes to match at the end of the column names.
        n (int): The target number used to find the column with the closest numeric prefix.

    Returns:
        list of str: A list of filtered column names. For each suffix in `suffixes`, the column
                     with the closest numeric prefix to `n` is included (if a match exists).

    Notes:
        - A column name is considered a match if it follows the pattern `<number>_<suffix>`.
        - If multiple columns match a suffix, the one with the numeric prefix closest to `n` is selected.
        - If no columns match a suffix, that suffix is skipped.

    Examples:
        columns = ["1_median", "5_mean", "10_std"]
        suffixes = ["median", "std"]
        n = 4
        Output: ["1_median", "10_std"]
    """
    filtered_columns = []
    
    for suffix in suffixes:
        matching_columns = [(col, int(re.match(rf'^(\d+)_({suffix})$', col).group(1))) 
                            for col in columns if re.match(rf'^\d+_{suffix}$', col)]
        
        if matching_columns:
            closest_column = min(matching_columns, key=lambda x: abs(x[1] - n))[0]
            filtered_columns.append(closest_column)
    
    return filtered_columns

def find_distribution(df, col):
    min_val = df[col].min()
    max_val = df[col].max()

    if min_val == 0:
        exponent_min = 0
    else:
        exponent_min = np.floor(np.log10(abs(min_val))).astype(int)
    if max_val == 0:
        exponent_max = 0
    else:
        exponent_max = np.floor(np.log10(abs(max_val))).astype(int)

    bases = np.linspace(1, 9, 9)
    exponents = list(range(exponent_min, exponent_max + 1))
    data = []
    for exponent in exponents:
        for base in bases:
            start_value = base * 10.0 ** exponent
            end_value = (base + 1) * 10.0 ** exponent
            
            filtered_df = df[(df[col] >= start_value) & (df[col] < end_value)]
            
            mean = filtered_df[col].mean() if not filtered_df.empty else np.nan
            std = filtered_df[col].std() if not filtered_df.empty else np.nan
            data.append({'base': int(base), 'exponent': int(exponent), 'mean': mean, 'std': std})
    
    return pd.DataFrame(data)

def find_distribution_only_exp(df, col):
    min_val = df[col].min()
    max_val = df[col].max()

    if min_val == 0:
        exponent_min = 0
    else:
        exponent_min = np.floor(np.log10(abs(min_val))).astype(int)
    if max_val == 0:
        exponent_max = 0
    else:
        exponent_max = np.floor(np.log10(abs(max_val))).astype(int)

    exponents = list(range(exponent_min, exponent_max + 1))
    data = []
    for exponent in exponents:
        start_value = 10.0 ** exponent
        end_value = 10.0 ** (exponent+1)
        
        filtered_df = df[(df[col] >= start_value) & (df[col] < end_value)]
        
        mean = filtered_df[col].mean() if not filtered_df.empty else np.nan
        std = filtered_df[col].std() if not filtered_df.empty else np.nan
        data.append({'exponent': int(exponent), 'mean': mean, 'std': std})
    
    return pd.DataFrame(data)

def apply_ranges(range_df, row):
    sigma = abs(row['combined_sigma']) 
    base = row['sigma_base']
    exponent = row['sigma_exponent']

    temp = range_df[(range_df['base'] == base) & (range_df['exponent'] == exponent)]
    
    if not temp.empty:
        mean = temp['mean'].values[0]
        std = temp['std'].values[0]

        return mean - 3 * std <= sigma <= mean + 3 * std
    return False 

def apply_ranges_only_exponent(range_df, row):
    sigma = abs(row['combined_sigma']) 
    exponent = row['sigma_exponent']

    temp = range_df[(range_df['exponent'] == exponent)]
    
    if not temp.empty:
        mean = temp['mean'].values[0]
        std = temp['std'].values[0]

        return mean - 3 * std <= sigma <= mean + 3 * std
    return False 

#############CROPPING, NOISE ADDITION AND DATA AUGMENTATIONS###################################################
###############################################################################################################
###############################################################################################################

#######################ORIGINAL DATA AUGMENTATION#################################################
def poisson_noise(img_data, ratio=0.5, ex_time=1, ron=3, dk=7, save=False,
                   sv_name=None, path=None):
    """
    Adds simulated Poisson noise, readout noise, and dark current noise to image data.

    Args:
        img_data (numpy.ndarray): The input image data to which noise is added.
                                  Should be a 2D array with at least two rows.
        ratio (float, optional): The scaling factor for exposure time. Defaults to 0.5.
        ex_time (float, optional): The exposure time for the image in seconds. Defaults to 1.
        ron (float, optional): The readout noise (standard deviation). Defaults to 3.
        dk (float, optional): The dark current in electrons per pixel. Defaults to 7.
        save (bool, optional): Whether to save the noisy image as a FITS file. Defaults to False.
        sv_name (str, optional): The name to use when saving the FITS file. Required if `save` is True.
        path (str, optional): The directory path to save the FITS file. Required if `save` is True.

    Returns:
        numpy.ndarray: The noisy image data. Returns `None` if the input data is invalid,
                       the exposure time is zero, or the shape of the image is invalid.

    Notes:
        - Poisson noise is simulated based on the scaled image data (`img * ratio`).
        - Dark current noise is modeled as Gaussian noise with a variance proportional to the
          dark current (`dk`) and the scaled exposure time (`time_ratio`).
        - Readout noise is modeled as Gaussian noise with a standard deviation equal to `ron`.

    Examples:
        # Adding noise to an image and returning it
        noisy_image = poisson_noise(img_data, ratio=0.8, ex_time=10, ron=5, dk=10)

        # Adding noise and saving the result to a file
        poisson_noise(img_data, ratio=0.8, ex_time=10, save=True, sv_name="noisy_image", path="./")

    Warnings:
        - If `save` is True, `sv_name` and `path` must be specified.
        - Pixels with zero values in `img_data` will remain zero in the noisy output.
    """

    if img_data is None or img_data.shape[0] < 2:
        return None

    height, width = img_data.shape[:2]
    
    time_ratio = ex_time * ratio
    if time_ratio == 0:
        return None
    img = img_data * ex_time
    
    # Generate noise
    dark_current_noise = np.random.normal(0, np.sqrt(dk * time_ratio / (60 * 60)), (height, width))  
    readout_noise = np.random.normal(0, ron, (height, width))
    poisson_noise_img = np.random.poisson(img * ratio)
    
    noisy_img = (poisson_noise_img + readout_noise + dark_current_noise) / time_ratio
    noisy_img[img_data == 0.0] = 0.0
    
    if save:
        save_fits(noisy_img, sv_name, path)
    return noisy_img

def black_level(H, W, out, ps=256, steps=100):
    """
    Identifies a patch within an image with the least amount of black pixels (zero values).
    This function is used in the dataset with original images. 

    Args:
        H (int): The height of the image.
        W (int): The width of the image.
        out (numpy.ndarray): The image data from which the patch is selected.
                             It should be a 2D or 3D array with shape (H, W).
        ps (int, optional): The size of the patch to consider, in pixels. Defaults to 256.
        steps (int, optional): The number of random steps (patches) to sample. Defaults to 100.

    Returns:
        numpy.ndarray: The patch with the least amount of black pixels (zeros).
                       Returns `None` if the image dimensions are smaller than the patch size
                       or the patch size is zero.

    Notes:
        - The function randomly samples `steps` number of patches of size `ps x ps` from the input image `out`.
        - It counts the number of zero values (black pixels) in each patch and selects the one with the least zeros.
        - The resulting patch is returned with any negative values clipped to zero.

    Examples:
        # Extract a patch with the least black pixels
        patch = black_level(1024, 1024, img_data, ps=256, steps=50)

    Warnings:
        - If the image is smaller than the patch size (`ps`), the function will return `None`.
    """
    if H < ps or W < ps or ps == 0:
        return None
    xx = np.random.randint(0, H - ps, steps)
    yy = np.random.randint(0, W - ps, steps)

    patch_area = ps * ps
    best_idx = 0

    patches = [out[x:x + ps, y:y + ps] for x, y in zip(xx, yy)]
    patches = np.array(patches)
    zero_counts = np.sum(patches == 0, axis=(1, 2))
    zero_percents = zero_counts / patch_area
    best_idx = np.argmin(zero_percents)
    xx = xx[best_idx]
    yy = yy[best_idx]
    out = out[xx:xx+ps, yy:yy+ps]
    return np.maximum(out, 0.0)

def out_in_image(gt_patch, func, kwargs_data):
    """
    Processes a ground truth image patch using a specified function and returns both the 
    ground truth patch and the processed output patch. Only used with original images

    Args:
        gt_patch (numpy.ndarray): The ground truth image patch to be processed.
        func (callable): A function to process the `gt_patch`. It should accept the patch 
                         and any additional arguments from `kwargs_data`.
        kwargs_data (dict): A dictionary of keyword arguments to pass to the processing function `func`.

    Returns:
        tuple: A tuple containing the processed ground truth patch and the processed output patch.
               Both patches are returned with an additional singleton dimension (i.e., reshaped to (H, W, 1)).
               Returns `None` if `gt_patch` is `None` or if an error occurs during processing.

    Notes:
        - The function assumes that `gt_patch` and the output of `func` are 2D arrays (grayscale images).
        - Both patches are expanded along the last axis (i.e., a channel dimension is added) to make them 3D arrays.
        - The function logs any errors that occur during the execution of the processing function.

    Example:
        # Apply a custom noise function to a ground truth patch
        gt_patch, in_patch = out_in_image(gt_image, poisson_noise, {'ratio': 0.5, 'ex_time': 1})
    """
    if gt_patch is None:
        return
    try:        
        in_patch = func(gt_patch, **kwargs_data)
        gt_patch = np.expand_dims(gt_patch, axis=(-1))
        in_patch = np.expand_dims(in_patch, axis=(-1))
    except Exception as err:
        logging.warning(f'an error occurred while image handling: {err}')
        return

    return gt_patch, in_patch

def data_augment(images, kwargs_data):
    """
    Performs data augmentation on a list of image file paths by processing each image through 
    a series of transformations, including reading the image, applying noise, and adjusting its 
    black level. The function yields augmented image patches for training or further processing.

    Args:
        images (list of str): A list of file paths to the images that need to be augmented.
        kwargs_data (dict): A dictionary of parameters controlling the augmentation process.
            It must contain:
                - 'start' (int): The starting ratio (default is 2).
                - 'stop' (int): The stopping ratio (default is 6).
                - 'ps' (int): Patch size (default is 256).
                - 'steps' (int): Number of steps for black level adjustment (default is 256).
                - 'type_of_image' (str): Type of image ('SCI' by default).
                - 'high' (int): Maximum intensity for image processing (default is 10000).
                - 'low' (int): Minimum intensity for image processing (default is 60).
                
    Yields:
        tuple: A tuple containing:
            - `in_patch` (numpy.ndarray): The augmented input image patch with added noise.
            - `gt_patch` (numpy.ndarray): The ground truth patch after noise addition.
            - A list with the `filepath` of the processed image.

    Raises:
        ValueError: If `start` is greater than or equal to `stop`.
        Exception: If an error occurs while reading or processing an image.

    Notes:
        - The function assumes the images are in FITS format, processed by `open_fits`.
        - The `poisson_noise` function is applied to introduce noise into the images.
        - The `black_level` function is used to adjust black levels by selecting patches of a specified size.
        - Only valid patches are yielded, and any files that cause errors during processing are skipped.

    Example:
        # Apply data augmentation to a list of images with specific parameters
        augmented_data = data_augment(images, {
            'start': 2, 'stop': 6, 'ps': 256, 'steps': 256, 
            'type_of_image': 'SCI', 'high': 10000, 'low': 60
        })
    """
        
    kwargs_data_ = {k:v for k, v in kwargs_data.items()}

    start = kwargs_data.get('start', 2)
    stop = kwargs_data.get('stop', 6)
    ps = kwargs_data_.get("ps", 256)
    steps = kwargs_data_.get("steps", 256)
    type_of_image = kwargs_data_.get('type_of_image', 'SCI')
    high = kwargs_data_.get('high', 10000)
    low = kwargs_data_.get('low', 60)
    del kwargs_data_['start']
    del kwargs_data_['stop']
    del kwargs_data_['ps']
    del kwargs_data_['steps']
    del kwargs_data_['high']
    del kwargs_data_['low']
    del kwargs_data_['type_of_image']

    if start >= stop:
        raise ValueError("`start` must be less than `stop`.")
    data = [(filepath, i) for filepath in images for i in range(start, stop)]
    random.shuffle(data)
    
    for filepath, ratio_ in data:
        try:
            result = open_fits(filepath, ratio_, type_of_image=type_of_image,
                                low=low, high=high)
            if result is None:
                #logging.warning(f"Skipping file {filepath} due to invalid result.")
                continue
            
            out, filepath, ex_time, ratio_ = result
            gt_patch = black_level(out.shape[0], out.shape[1], out, ps=ps, steps=steps)
            kwargs_data_['ratio'] = ratio_
            kwargs_data_['ex_time'] = ex_time
            result = out_in_image(gt_patch=gt_patch, func=poisson_noise, 
                                  kwargs_data=kwargs_data_)
            if result is None:
                logging.warning(f"Skipping file {filepath} due to error in out_in_image processing.")
                continue
            
            gt_patch, in_patch = result
            yield (in_patch, gt_patch, [filepath])

        except Exception as err:
            logging.error(f"An error occurred while processing file {filepath}: {err}")
            continue

####### NEW DATA AUGMENTATION ################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################

#################WITH FIT#########################################
##############################################################################
def prepare_data(sampled_data, fit_data, training, data_folder):
    """
    Prepares data for training or evaluation by reading the corresponding FITS files, applying noise, 
    and returning the data with the noise added. This function supports both training and evaluation modes.
    The noise is calculated based on a fit between exposure time and the standard deviation of the background noise
    This function has been originally used to calculate the variances of the background with its main function 
    'data_augment_2()'. However, the fit turned out to be inconclusive so 'prepare_data_2()' and 'data_augment_3()' are
    used in their stead.
    
    Args:
        sampled_data (pandas.DataFrame): A DataFrame containing information about the sampled data,
            which includes file names and other metadata (e.g., `name`, `exp_time`, `sigma`).
        fit_data (pandas.DataFrame): A DataFrame containing fit parameters (`t`, `a`, `dt`, `da`) 
            used to calculate noise characteristics.
        training (bool): A flag indicating whether the data is for training (`True`) or evaluation (`False`).
        data_folder (str): The folder where the FITS files are stored. The folder structure is expected 
            to have subdirectories for 'training' and 'eval'.

    Yields:
        tuple: A tuple containing:
            - `with_noise` (numpy.ndarray): The noisy image data, with added noise based on calculated sigma.
            - `file` (numpy.ndarray): The original image data (ground truth).
            - A list with the generated file name, including the noise applied (e.g., `filepath_sigma_value.fits`).
    
    Raises:
        Exception: If there is an error in processing the data (e.g., if a FITS file cannot be opened).

    Notes:
        - The function uses the `open_fits_2` function to read FITS files from disk.
        - Noise is added to the image using a normal distribution with a calculated standard deviation (`sigma_kernel`), 
          which is based on the parameters from `fit_data`.
        - The noise level (`sigma_kernel`) is calculated using a power law function.
        - The resulting noisy image and ground truth data are returned in the format required for training or evaluation.
    
    Example:
        # Prepare training data for a set of sampled images
        for noisy_data, ground_truth, file_name in prepare_data(sampled_data, fit_data, True, 'data'):
            # Process noisy data and ground truth
            pass
    """
    names = sampled_data['name'].tolist()
    data = {}
    for index, row in sampled_data.iterrows():
        names.pop(0)
        filepath = row['name']
        file = None
        if filepath in data:
            file = data[filepath]
        else:
            if training == False:
                train_eval = 'eval'
            else:
                train_eval = 'training'

            actual_filepath = f'{os.path.dirname(__file__)}/{data_folder}/{train_eval}/{filepath}'
            file = open_fits_2(actual_filepath)
            if file is not None:
                file = np.nan_to_num(file, nan=0)
                if filepath in names:
                    data[filepath] = file

        t = fit_data['t'].values[0]
        a = fit_data['a'].values[0]
        dt = np.random.uniform(-fit_data['dt'].values[0], fit_data['dt'].values[0])
        da = np.random.uniform(-fit_data['da'].values[0], fit_data['da'].values[0])
        t += dt
        a += da
        sigma_2 = power_law(row['exp_time'], t, a)
        sigma_kernel_2 = abs(sigma_2 - row['sigma']**2)
        sigma_kernel = np.sqrt(sigma_kernel_2)
        with_noise = file + np.random.normal(loc=0, scale=sigma_kernel, size=file.shape)
        yield (np.expand_dims(with_noise, -1), np.expand_dims(file, -1), [f"{filepath.split('.fits')[0]}_{sigma_kernel_2:.5f}".replace('.', '_') + ".fits"])

def data_augment_2(images, kwargs_data):
    """
    Perform data augmentation by preparing a dataset with added noise for training or evaluation. This function 
    reads metadata and fit data, filters it according to certain conditions, and samples images accordingly 
    before applying noise using the `prepare_data` function.

    Args:
        images (list): A list of image file paths to be processed. Not directly used within the function but provided for compatibility.
        kwargs_data (dict): A dictionary containing various parameters for the data augmentation process:
            - 'metadata_filepath' (str): Path to the CSV file containing metadata information (including exposure time, sigma, and location).
            - 'exposure_col' (str): The column name in the metadata CSV file for exposure time.
            - 'name_col' (str): The column name in the metadata CSV file for image names.
            - 'sigma_col' (str): The column name in the metadata CSV file for sigma values.
            - 'fit_data_filepath' (str): Path to the CSV file containing the fit data.
            - 'fit_data_cols' (list): A list of the expected column names in the fit data CSV file.
            - 'location_col' (str): The column name in the metadata CSV file for location information (e.g., 'training' or 'eval').
            - 'abs_mean_col' (str): The column name in the metadata CSV file for the absolute mean value.
            - 'samples' (int): The number of samples to be selected for augmentation.
            - 'steps' (int): The step size to decrement the exposure time during the augmentation process.
            - 'low' (int): The lower bound for valid exposure times.
            - 'high' (int): The upper bound for valid exposure times.
            - 'training' (bool): Whether the dataset is for training (`True`) or evaluation (`False`).
            - 'data_folder' (str): The folder where the image data is stored.

    Yields:
        tuple: A tuple containing:
            - `with_noise` (numpy.ndarray): The noisy image data generated using the parameters provided.
            - `file` (numpy.ndarray): The original image data (ground truth).
            - A list with the generated file name, including the noise applied (e.g., `filepath_sigma_value.fits`).
    
    Raises:
        Exception: If any of the required parameters are missing or if the metadata or fit data file paths are invalid.
    
    Notes:
        - The function reads and filters metadata based on exposure time and location for training or evaluation.
        - The fit data is used to calculate the noise levels, which are applied to the image data.
        - Images are sampled based on the `samples` parameter and the absolute mean (`abs_mean`), sorted in descending order.
        - The `prepare_data` function is called for each batch of selected data to apply noise and yield augmented data.

    Example:
        # Augment data for training images
        for noisy_data, ground_truth, file_name in data_augment_2(images, kwargs_data):
            # Process noisy data and ground truth
            pass
    """
    metadata_filepath = kwargs_data.get('metadata_filepath', None)
    exposure_col = kwargs_data.get('exposure_col', None)
    name_col = kwargs_data.get("name_col", None)
    sigma_col = kwargs_data.get("sigma_col", None)
    fit_data_filepath = kwargs_data.get('fit_data_filepath', None)
    fit_data_cols = kwargs_data.get('fit_data_cols', None)
    location_col = kwargs_data.get("location_col", None)
    abs_mean_col = kwargs_data.get("abs_mean_col", None)
    samples = kwargs_data.get('samples', 5000)
    val_samples = kwargs_data.get('val_samples', 5000)
    steps = kwargs_data.get("steps", 100)
    low = kwargs_data.get("low", 90)
    high = kwargs_data.get('high', 15000)
    training = kwargs_data.get('training', True), 
    data_folder = kwargs_data.get('data_folder', None) 

    columns_to_check = [
        metadata_filepath,
        exposure_col,
        name_col,
        sigma_col,
        fit_data_filepath,
        fit_data_cols, 
        location_col,
        abs_mean_col,
        data_folder
    ]

    if any(col is None for col in columns_to_check):
        raise Exception(f'One of the important parameters in the data arguments is None: {kwargs_data}. EXITING')
    if not os.path.exists(metadata_filepath):
        raise Exception(f'{metadata_filepath} does not exist. EXITING')
    if not os.path.exists(fit_data_filepath):
        raise Exception(f'{fit_data_filepath} does not exist. EXITING')
    
    df = pd.read_csv(metadata_filepath)
    fit_data = pd.read_csv(fit_data_filepath)
    
    if set(fit_data.columns) != set(fit_data_cols):
        raise Exception(f'{fit_data.columns} are not the required columns for the approximation: {fit_data_cols}')
    if exposure_col not in df or name_col not in df or sigma_col not in df:
        raise Exception(f'{exposure_col} or {name_col} or {sigma_col} not in the metadata. EXITING')
    
    info = []
    for index, row in df.iterrows():
        exp = row[exposure_col]
        while True:
            exp -= steps
            if  exp > low and exp < high:
                info.append({'name' : row[name_col], 'exp_time' : exp, 'sigma' : row[sigma_col],
                 'location':row[location_col], 'abs_mean':row[abs_mean_col]})
            else:
                break
    info = pd.DataFrame(info)
    info = info.dropna(subset=['sigma', 'exp_time', 'location', 'name', 'abs_mean'])
    info = info[(info['sigma'] != 0) | (info['exp_time'] != 0)]
    info['sigma'] = info['sigma'].abs()
    info['exp_time'] = info['exp_time'].abs()

    if training:
        info = info[info['location'].apply(lambda x: 'training' in str(x) if pd.notnull(x) else False)]
    else:
        info = info[info['location'].apply(lambda x: 'eval' in str(x) if pd.notnull(x) else False)]
    info.reset_index(inplace=True, drop=True)
    
    if len(info) and samples > len(info):
        x = len(info)
    elif len(info):
        x = samples
    else:
        raise Exception(f'no adaquate sampling is possible')
    
    info = info.sample(frac=1).reset_index(drop=True)
    info.sort_values(by='abs_mean', ascending=False, inplace=True)
    sampled = info.head(x).reset_index(drop=True)
    sampled = sampled.sample(frac=1).reset_index(drop=True)
    yield from prepare_data(sampled, fit_data, training, data_folder)

######################WITHOUT FIT##################################
###################SCALING#######################
def min_max_normalization(data):
    min_val, max_val = np.min(data), np.max(data)
    if (max_val - min_val) == 0:
        return None
    return (data - min_val) / (max_val - min_val), min_val, max_val

def inverse_min_max_normalization(normalized_data, min_val, max_val):
    return (normalized_data * (max_val - min_val)) + min_val

def zscore_normalization(data):
    mean, std = np.mean(data), np.std(data)
    if std == 0:
        return None
    return (data - mean) / std, mean, std

def inverse_zscore_normalization(normalized_data, mean, std):
    if std == 0:
        return None
    return (normalized_data * std) + mean

def adaptive_log_transform_and_normalize(data):
    # Shift to avoid negative values
    min_data = np.min(data)
    shift_value = -min_data + 1 if min_data < 0 else 1
    log_data = np.log(data + shift_value)
    # Min-Max normalization
    min_val = np.min(log_data)
    max_val = np.max(log_data)
    # Prevent division by zero if min_val == max_val
    if min_val == max_val:
        return None, None, None, None
    normalized_data = (log_data - min_val) / (max_val - min_val)
    return normalized_data.astype(np.float32), min_val, max_val, shift_value

def inverse_adaptive_log_transform_and_denormalize(predicted_data, original_min_val, original_max_val, shift_value):
    if original_min_val is None or original_max_val is None:
        return None
    # Reverse Min-Max normalization
    log_data = predicted_data * (original_max_val - original_min_val) + original_min_val
    # Reverse log transformation
    denormalized_data = np.exp(log_data) - shift_value
    return denormalized_data.astype(np.float32)

def is_relevant_crop(stats, delta=0.5):
    """
    Determines if a crop is considered relevant based on its absolute mean value relative to the original uncropped image.

    This function compares the absolute mean of the cropped image (`abs_mean`) with a threshold, which is defined
    as a fraction (`delta`) of the absolute mean of the original uncropped image (`org_abs_mean`). If the absolute
    mean of the cropped image exceeds the threshold, the crop is considered relevant.

    Args:
        stats (dict): A dictionary containing the following key-value pairs:
            - 'org_abs_mean' (float): The absolute mean of the original uncropped image.
            - 'abs_mean' (float): The absolute mean of the cropped image.
        delta (float, optional): The fraction of the original absolute mean to be used as the threshold. Defaults to 0.5.

    Returns:
        bool: `True` if the crop is relevant (i.e., its absolute mean is greater than the threshold), `False` otherwise.

    Example:
        stats = {'org_abs_mean': 100, 'abs_mean': 60}
        result = is_relevant_crop(stats, delta=0.5)
        print(result)  # Output: True, since 60 > 100 * 0.5

    Notes:
        - The `delta` parameter allows for flexibility in how strict the threshold is.
        - If the absolute mean of the cropped image is smaller than the threshold, the crop is considered not relevant.
    """
        
    abs_uncropped = stats['org_abs_mean']
    abs_crop = stats['abs_mean']
    abs_threshold = abs_uncropped * delta
    return abs_crop > abs_threshold

def filtering_df(data, n_samples, delta, id_, abs_, base, exponent, noise_ratio):
    """
    Filters and selects a subset of rows from a DataFrame based on relevance and specific sorting criteria.
    The main aim of the function is to guarantee a good selection. This is achieved first by grouping by dataset id
    and selecting the highest absolute mean value of the cropped image, the highest base and the lowest exponent difference
    between the noisy image and the original image given by 'exponent' and then adding these rows to a new list and removing
    such rows from the main DataFrame. This will be repeated till'n_samples' are achieved. 

    The function applies a relevance check to each row using the `is_relevant_crop` function, filters the rows that
    are considered relevant, and then selects a subset of rows based on specified sorting and grouping criteria.

    The selection process groups the data by the specified `id_` column, drops duplicate entries based on the `abs_`
    column, and selects rows with the highest `abs_` values, while considering other factors such as `base` and 
    `exponent`. If the desired number of samples (`n_samples`) is not achievable, it adjusts by selecting the maximum
    number of samples possible.

    Args:
        data (pd.DataFrame): The input DataFrame containing the data to be filtered. Must contain the specified
            columns for filtering and sorting.
        n_samples (int): The desired number of samples to select. If there are fewer relevant rows than `n_samples`,
            the function will return as many rows as possible.
        delta (float): A threshold factor for the `is_relevant_crop` function to determine if a row is relevant.
        id_ (str): The column name representing the identifier used for grouping the data (dataset ID).
        abs_ (str): The column name representing the absolute mean values used for sorting and filtering.
        base (str): The column name representing the base value used for sorting (base of the variance in form of b*10**N).
        exponent (str): The column name representing the exponent value used for sorting (N in b*10**N).

    Returns:
        pd.DataFrame: A filtered DataFrame containing the selected rows, with the specified number of samples.
            The DataFrame is sorted by `abs_` in descending order, with duplicates removed based on `abs_`, and grouped
            by the `id_` column.

    Example:
        data = pd.DataFrame({
            'id': [1, 2, 1, 2],
            'abs_mean': [0.8, 0.9, 0.75, 0.85],
            'base': [0.5, 0.6, 0.5, 0.6],
            'exponent': [2, 3, 2, 3],
            'org_abs_mean': [1, 1, 1, 1],
            'sigma': [0.1, 0.2, 0.15, 0.2]
        })
        
        filtered_data = filtering_df(data, n_samples=2, delta=0.5, id_='id', abs_='abs_mean', base='base', exponent='exponent')
        print(filtered_data)

    Notes:
        - The function will first filter out rows that are not relevant based on the `delta` threshold.
        - If there are fewer rows than `n_samples`, the function will return as many as possible, with duplicates removed.
        - The resulting DataFrame is sorted by `abs_` in descending order, `base` and `exponent` are sorted according to
          their specified ascending/descending order.
    """
    if n_samples >= len(data):
        return data

    data['valid'] = data.apply(lambda row: is_relevant_crop(row, delta=delta), axis=1)
    data = data[data['valid']]
    if n_samples >= len(data):
        return data
    old_n_samples = n_samples
    if len(data) >= 2*n_samples:
        n_samples *= 2

    data.loc[:, 'temp_index'] = np.arange(0, len(data))
    dfs = []
    already_in = 0
    while True:
        n_samples -= already_in
        sorted_df = data.sort_values(by=[id_, abs_, base, exponent], ascending=[True, False, False, True])
        #sorted_df[[id_, abs_, base, exponent, 'combined_sigma', 'sigma']].to_csv('sorted.csv', index=False)

        result = sorted_df.groupby(id_).apply(
            lambda group: group.drop_duplicates(subset=abs_, keep='first')
        )
        if len(result) >= n_samples:
            dfs.append(result.sort_values(by=[abs_], ascending=False, ignore_index=True).iloc[:n_samples])
            break
        else:
            dfs.append(result)
            data = data[~data['temp_index'].isin(result['temp_index'])]
            already_in += len(result)
            if data.empty:
                break
    
    return pd.concat(dfs, ignore_index=True).sort_values(by=[noise_ratio], ascending=True).iloc[:old_n_samples]

def filtering_df_5(data, n_samples, delta, id_, abs_, base, exponent):
    """
    Filters and selects a subset of rows from a DataFrame based on relevance and specific sorting criteria.
    The main aim of the function is to guarantee a good selection. This is achieved first by grouping by dataset id
    and selecting the highest absolute mean value of the cropped image, the highest base and the lowest exponent difference
    between the noisy image and the original image given by 'exponent' and then adding these rows to a new list and removing
    such rows from the main DataFrame. This will be repeated till'n_samples' are achieved. 

    The function applies a relevance check to each row using the `is_relevant_crop` function, filters the rows that
    are considered relevant, and then selects a subset of rows based on specified sorting and grouping criteria.

    The selection process groups the data by the specified `id_` column, drops duplicate entries based on the `abs_`
    column, and selects rows with the highest `abs_` values, while considering other factors such as `base` and 
    `exponent`. If the desired number of samples (`n_samples`) is not achievable, it adjusts by selecting the maximum
    number of samples possible.

    Args:
        data (pd.DataFrame): The input DataFrame containing the data to be filtered. Must contain the specified
            columns for filtering and sorting.
        n_samples (int): The desired number of samples to select. If there are fewer relevant rows than `n_samples`,
            the function will return as many rows as possible.
        delta (float): A threshold factor for the `is_relevant_crop` function to determine if a row is relevant.
        id_ (str): The column name representing the identifier used for grouping the data (dataset ID).
        abs_ (str): The column name representing the absolute mean values used for sorting and filtering.
        base (str): The column name representing the base value used for sorting (base of the variance in form of b*10**N).
        exponent (str): The column name representing the exponent value used for sorting (N in b*10**N).

    Returns:
        pd.DataFrame: A filtered DataFrame containing the selected rows, with the specified number of samples.
            The DataFrame is sorted by `abs_` in descending order, with duplicates removed based on `abs_`, and grouped
            by the `id_` column.

    Example:
        data = pd.DataFrame({
            'id': [1, 2, 1, 2],
            'abs_mean': [0.8, 0.9, 0.75, 0.85],
            'base': [0.5, 0.6, 0.5, 0.6],
            'exponent': [2, 3, 2, 3],
            'org_abs_mean': [1, 1, 1, 1],
            'sigma': [0.1, 0.2, 0.15, 0.2]
        })
        
        filtered_data = filtering_df(data, n_samples=2, delta=0.5, id_='id', abs_='abs_mean', base='base', exponent='exponent')
        print(filtered_data)

    Notes:
        - The function will first filter out rows that are not relevant based on the `delta` threshold.
        - If there are fewer rows than `n_samples`, the function will return as many as possible, with duplicates removed.
        - The resulting DataFrame is sorted by `abs_` in descending order, `base` and `exponent` are sorted according to
          their specified ascending/descending order.
    """
    if n_samples >= len(data):
        return data

    data['valid'] = data.apply(lambda row: is_relevant_crop(row, delta=delta), axis=1)
    data = data[data['valid']]
    data['ratio'] = data['abs_mean']/(np.maximum(data['org_abs_mean'], 1e-7))
    if n_samples >= len(data):
        return data

    data.loc[:, 'temp_index'] = np.arange(0, len(data))
    dfs = []
    already_in = 0
    first = True
    while True:
        n_samples -= already_in
        if first:
            sorted_df = data.sort_values(by=[id_, 'ratio', base, exponent], ascending=[True, False, False, True])
        else:
            data = data.sample(frac=1)
            sorted_df = data.sort_values(by=[id_, base, exponent], ascending=[True, False, False, True])

        result = sorted_df.groupby(id_).apply(
            lambda group: group.drop_duplicates(subset=abs_, keep='first')
        )
        if len(result) >= n_samples:
            dfs.append(result.sort_values(by=['ratio'], ascending=False, ignore_index=True).iloc[:n_samples])
            break
        else:
            dfs.append(result)
            data = data[~data['temp_index'].isin(result['temp_index'])]
            already_in += len(result)
            if data.empty:
                break
        first = False
    return pd.concat(dfs, ignore_index=True).sample(frac=1)

########################BASED ON VARIANCES ##################################
        
def augment_samples(sigma, lowest_power=-4, highest_power=5, n_samples_per_magnitude=3):
    """
    Generates a list of augmented variance values based on the input standard deviation (`sigma`).

    The function computes the variance based on the provided `sigma`, then generates multiple augmented variance 
    values by modifying the base and exponent of the logarithmic representation of the variance. The generated 
    variances are spread across a specified range of powers, with a defined number of samples for each magnitude.

    The augmentations are produced for powers of 10 from the logarithmic scale of the variance, within the range 
    from `lowest_power` to `highest_power`. For each magnitude, the function generates values based on evenly spaced 
    numbers and returns a list of tuples containing the variance, the base factor used, and the shift in exponent.

    Args:
        sigma (float): The standard deviation of the data, used to compute the variance. It must not be zero or NaN.
        lowest_power (int, optional): The lowest exponent (log10 scale) to consider for augmentation. Defaults to -4.
        highest_power (int, optional): The highest exponent (log10 scale) to consider for augmentation. Defaults to 5.
        n_samples_per_magnitude (int, optional): The number of augmented samples to generate for each magnitude 
            (between `lowest_power` and `highest_power`). Defaults to 3.

    Returns:
        list of tuples: A list of augmented variance values represented as tuples, where each tuple contains:
            - The augmented variance (float).
            - The base factor (int) used for the variance calculation.
            - The shift in exponent (int).

    Example:
        augmented_variances = augment_samples(0.1, lowest_power=-3, highest_power=3, n_samples_per_magnitude=5)
        print(augmented_variances)

    Notes:
        - The function uses logarithmic scaling to adjust the variance and generates augmented values within the 
          specified range of exponents (`lowest_power` to `highest_power`).
        - The variance is augmented by modifying the base value for each power of 10, sampled evenly in the range 
          between 1 and 9 for each magnitude.
        - The function ensures that only valid values of `sigma` (non-zero and non-NaN) are considered for augmentation.
    """
    variances = []
    if is_not_nan(sigma) and sigma != 0.0:
        var = sigma**2

        exponent = np.floor(np.log10(abs(var))).astype(int)
        base = np.floor(var / (10.0**exponent))
        
        if lowest_power <= exponent <= highest_power:
            for x in range(exponent, highest_power + 1):
                start = base if x == exponent else 1
                stop = 9
                for b in evenly_spaced_numbers(start, stop, n_samples_per_magnitude):
                    variance = b * 10.0**x
                    std = np.sqrt(variance)
                    variances.append((variance, b, exponent, x-exponent))
    return variances

def prepare_data_2(sampled_data, training, data_folder, scaling=None):
    """
    The default function used in the augmentation in 'data_augment_3()'
    Prepares noisy and original image pairs for training or evaluation by applying noise to the images based on the 
    provided noisy sigma values. 

    The function loads FITS images from the specified directory, adds noise to the images based on the provided 
    noisy sigma values, and then yields pairs of noisy and clean images along with their modified filenames. 
    This function is designed to handle both training and evaluation data, based on the `training` parameter. 

    Args:
        sampled_data (pd.DataFrame): A dataframe containing information about the images to be processed, 
            including the filenames (`name`) and the noisy sigma values (`noisy_sigma`).
        training (bool): A boolean indicating whether to prepare data for training (`True`) or evaluation (`False`).
        data_folder (str): The path to the folder containing the image files.

    Yields:
        tuple: A tuple containing:
            - Noisy image (np.ndarray): The image with added Gaussian noise based on the noisy sigma value.
            - Clean image (np.ndarray): The original (clean) image.
            - str: A modified filename for the noisy image, which includes the squared noisy sigma value.

    Example:
        for noisy, clean, filename in prepare_data_2(sampled_data, True, "/data/images"):
            # Use noisy, clean, filename as needed for training
            pass

    Notes:
        - The function assumes that the image files are in FITS format and uses the `open_fits_2` function to load 
          them.
        - The noise added to the images is Gaussian, with the standard deviation defined by `noisy_sigma`.
        - The noisy images are yielded with a modified filename that includes the squared noisy sigma value, which 
          is useful for identifying different noise levels.
        - The function ensures that the images are not duplicated in memory by tracking loaded images in a dictionary.
        - It also handles NaN values by replacing them with zeros before yielding the data.

    """
    if 'name' not in sampled_data or 'noisy_sigma' not in sampled_data:
        return

    count = 0
    while count < len(sampled_data):
        names = sampled_data['name'].tolist()
        data = {}

        for index, row in sampled_data.iterrows():
            if count >= len(sampled_data):
                break

            names.pop(0)
            filepath = row['name']
            file = None
            sigma_kernel = row['noisy_sigma']
            
            if filepath in data:
                file = data[filepath]
            else:
                if training == False:
                    train_eval = 'eval'
                else:
                    train_eval = 'training'

                actual_filepath = f'{os.path.dirname(__file__)}/{data_folder}/{train_eval}/{filepath}'
                file = open_fits_2(actual_filepath)
                if file is not None:
                    file = np.nan_to_num(file, nan=0)
                    file = np.maximum(file, 0)
                    if filepath in names and filepath not in data:
                        data[filepath] = file
            
            if filepath not in names and filepath in data:
                del data[filepath]
            if file is None:
                continue

            with_noise = file + np.random.normal(loc=0, scale=sigma_kernel, size=file.shape)
            with_noise = np.nan_to_num(with_noise, nan=0)
            min_val = np.min(with_noise)
            if min_val < 0:
                with_noise -= min_val

            if scaling is not None and scaling == 'log_min_max':
                result =  adaptive_log_transform_and_normalize(with_noise)
                if result is None:
                    continue
                with_noise, noise_min_val, noise_max_val, noise_shift = result

                result =  adaptive_log_transform_and_normalize(file)
                if result is None:
                    continue
                file, org_min_val, org_max_val, org_shift = result

                stats = [f"{filepath.split('.fits')[0]}_{sigma_kernel:.5f}".replace('.', '_') + ".fits", 
                        str(org_min_val), str(org_max_val), str(org_shift), str(noise_min_val), str(noise_max_val), str(noise_shift)]
                
            elif scaling is not None and scaling == 'z_scale':
                result =  zscore_normalization(file)
                if result is None:
                    continue
                file, org_mean, org_std = result
                result =  zscore_normalization(with_noise)
                if result is None:
                    continue
                with_noise, noise_mean, noise_std = result
                stats = [f"{filepath.split('.fits')[0]}_{sigma_kernel:.5f}".replace('.', '_') + ".fits", 
                        str(org_mean), str(org_std), str(noise_mean), str(noise_std)]
                
            elif scaling is not None and scaling == 'min_max':
                result =  min_max_normalization(with_noise)
                if result is None:
                    continue
                with_noise, noise_min, noise_max = result
                result =  min_max_normalization(file)
                if result is None:
                    continue
                file, org_min, org_max = result

                stats = [f"{filepath.split('.fits')[0]}_{sigma_kernel:.5f}".replace('.', '_') + ".fits", 
                        str(org_min), str(org_max), str(noise_min), str(noise_max)]
            else:
                stats = [f"{filepath.split('.fits')[0]}_{sigma_kernel:.5f}".replace('.', '_') + ".fits"]
            yield (np.expand_dims(with_noise, -1), np.expand_dims(file, -1), stats)
            count += 1

def prepare_data_3(sampled_data, training, data_folder, scaling=None):
    if 'name' not in sampled_data or 'noisy_sigma' not in sampled_data:
        return

    count = 0
    while count < len(sampled_data):
        names = sampled_data['name'].tolist()
        data = {}
        for index, row in sampled_data.iterrows():
            if count >= len(sampled_data):
                break

            names.pop(0)
            filepath = row['name']
            file = None
            noisy_sigma = row['noisy_sigma']
            exp_time = row['exp_time']
            new_exp_time = row['new_exp_time']
            exp_ratio = row['exp_ratio']
            sigma = row['sigma']
            median = row['median']

            if filepath in data:
                file = data[filepath]
            else:
                if training == False:
                    train_eval = 'eval'
                else:
                    train_eval = 'training'

                actual_filepath = f'{os.path.dirname(__file__)}/{data_folder}/{train_eval}/{filepath}'
                
                file = open_fits_2(actual_filepath)
                if file is not None:
                    if filepath in names and filepath not in data:
                        data[filepath] = file
                        
            if filepath not in names and filepath in data:
                del data[filepath]
            if file is None:
                continue
            file = np.maximum(file, 1e-8)
            file = np.nan_to_num(file, nan=0.0, posinf=0.0, neginf=0.0)
            #with_noise = create_noisy_image(file, exp_time, exp_ratio)
            with_noise = create_noisy_image_gaussian(file, exp_ratio, median, sigma)

            if scaling is not None and scaling == 'log_min_max':
                result =  adaptive_log_transform_and_normalize(with_noise)
                if result is None:
                    continue
                with_noise, noise_min_val, noise_max_val, noise_shift = result

                result =  adaptive_log_transform_and_normalize(file)
                if result is None:
                    continue
                file, org_min_val, org_max_val, org_shift = result

                stats = [f"{filepath.split('.fits')[0]}_{round(exp_time)}__{round(new_exp_time)}".replace('.', '_') + ".fits", 
                        str(org_min_val), str(org_max_val), str(org_shift), str(noise_min_val), str(noise_max_val), str(noise_shift)]
                
            elif scaling is not None and scaling == 'z_scale':
                result =  zscore_normalization(with_noise)
                if result is None:
                    continue
                with_noise, noise_mean, noise_std = result
                result =  zscore_normalization(file)
                if result is None:
                    continue
                file, org_mean, org_std = result

                stats = [f"{filepath.split('.fits')[0]}_{round(exp_time)}__{round(new_exp_time)}".replace('.', '_') + ".fits", 
                        str(org_mean), str(org_std), str(noise_mean), str(noise_std)]
                
            elif scaling is not None and scaling == 'min_max':
                result =  min_max_normalization(with_noise)
                if result is None:
                    continue
                with_noise, noise_min, noise_max = result
                result =  min_max_normalization(file)
                if result is None:
                    continue
                file, org_min, org_max = result

                stats = [f"{filepath.split('.fits')[0]}_{round(exp_time)}__{round(new_exp_time)}".replace('.', '_') + ".fits", 
                        str(org_min), str(org_max), str(noise_min), str(noise_max)]
            else:
                stats = [f"{filepath.split('.fits')[0]}_{round(exp_time)}__{round(new_exp_time)}".replace('.', '_') + ".fits"]
            yield (np.expand_dims(with_noise, -1), np.expand_dims(file, -1), stats)
            count += 1

def data_augment_3(images, kwargs_data, scaling=None):
    """
    This is the main augmentation function used. 
    Augments data samples based on the provided metadata and other augmentation parameters.
    
    This function reads metadata from a CSV file and processes each row to apply data augmentation 
    techniques to the images. For each valid row, it computes a range of synthetic variances based 
    on the provided noise characteristics, generates augmented data, and prepares it for further 
    processing. The function yields augmented data samples in batches, which can be used for 
    training or evaluation purposes.

    Args:
        images (list): A list of image file paths to be processed. These are used for sampling.
        kwargs_data (dict): A dictionary containing various parameters necessary for the augmentation 
                            process, including metadata filepath, column names, sample parameters, 
                            and thresholds for processing. The keys are:
                            - 'metadata_filepath' (str): Path to the CSV file containing metadata.
                            - 'exposure_col' (str): The name of the column containing exposure values.
                            - 'name_col' (str): The name of the column containing image file names.
                            - 'sigma_col' (str): The name of the column containing sigma values.
                            - 'location_col' (str): The name of the column indicating location (training/eval).
                            - 'abs_mean_col' (str): The column containing absolute mean values.
                            - 'org_abs_mean_col' (str): The original absolute mean column for the uncropped images.
                            - 'dataset' (str): The dataset identifier.
                            - 'samples' (int): Number of samples to generate.
                            - 'low' (float): Lower threshold for exposure values.
                            - 'high' (float): Upper threshold for exposure values.
                            - 'training' (bool): Whether to filter for training data (True) or evaluation (False).
                            - 'lowest_power' (int): The lowest power for the variance augmentation.
                            - 'highest_power' (int): The highest power for the variance augmentation.
                            - 'n_samples_per_magnitude' (int): Number of samples to generate per variance magnitude.
                            - 'delta' (float): Threshold for filtering data based on absolute mean values.
                            - 'conf_low' (int): Lower threshold for confidence index filtering.
                            - 'conf_up' (int): Upper threshold for confidence index filtering.
                            - 'data_folder' (str): Path to the data folder containing the images.

    Yields:
        tuple: A tuple containing augmented data:
            - Noisy image data (numpy array of shape (height, width, 1)).
            - Clean image data (numpy array of shape (height, width, 1)).
            - Image file name (string, e.g., "image_name_sigma_value.fits").
    
    Raises:
        Exception: If any required parameter in `kwargs_data` is missing or invalid, or if insufficient 
                   samples are available for the augmentation process.
    
    Example:
        # Example usage
        data_generator = data_augment_3(images=["image1.fits", "image2.fits"], kwargs_data=kwargs)
        for noisy_image, clean_image, filename in data_generator:
            # Use the augmented data in training or evaluation
            pass
    """
        
    metadata_filepath = kwargs_data.get('metadata_filepath', None)
    exposure_col = kwargs_data.get('exposure_col', None)
    name_col = kwargs_data.get("name_col", None)
    sigma_col = kwargs_data.get("sigma_col", None)
    location_col = kwargs_data.get("location_col", None)
    abs_mean_col = kwargs_data.get("abs_mean_col", None)
    org_abs_mean_col = kwargs_data.get("org_abs_mean_col", None)
    dataset = kwargs_data.get("dataset", None)
    median_col = kwargs_data.get("median_col", None)
    samples = kwargs_data.get('samples', 5000)
    val_samples = kwargs_data.get('val_samples', 5000)
    low = kwargs_data.get("low", 90)
    high = kwargs_data.get('high', 15000)
    training = kwargs_data.get('training', True)
    lowest_power = kwargs_data.get('lowest_power', -4)
    highest_power = kwargs_data.get('highest_power', 2)
    data_folder = kwargs_data.get('data_folder', None)
    n_samples_per_magnitude = kwargs_data.get('n_samples_per_magnitude', 2) 
    delta = kwargs_data.get('delta', 0.25) 
    #cutoff = kwargs_data.get('cutoff', 80)
    #suffixes = kwargs_data.get('suffixes', None)
    #n_sigma_stat = kwargs_data.get('cutoff', 3)
    interested_stats_col = kwargs_data.get('light_col', None)
    abs_percent = kwargs_data.get('abs_percent', 25)
    conf_low = kwargs_data.get('conf_low', 2)
    conf_up = kwargs_data.get('conf_up', 2)
    threshold= kwargs_data.get('threshold', 100)
    lowest_exposure = kwargs_data.get('lowest_exposure', 200)
    times = kwargs_data.get('times', 2)
    ratio = kwargs_data.get('ratio', 10)

    columns_to_check = [
        metadata_filepath,
        exposure_col,
        name_col,
        sigma_col,
        location_col,
        abs_mean_col,
        data_folder,
        median_col,
        dataset,
        org_abs_mean_col,
        interested_stats_col
        #suffixes
    ]
    if any(col is None for col in columns_to_check):
        raise Exception(f'One of the important parameters in the data arguments is None: {kwargs_data}. EXITING')
    conf_up = set_threshold(conf_up)
    conf_low = set_threshold(conf_low, start=True)

    global MEM_CACHED, MEM_CACHED_EVAL
    condition = MEM_CACHED is None if training else MEM_CACHED_EVAL is None
    if condition:
        df = pd.read_csv(metadata_filepath)
        df = df[(df[exposure_col] >= times*low) & (df[exposure_col] <= high)]

        #interested_stats_columns = find_columns(df.columns, suffixes, cutoff)
        #if not interested_stats_columns:
        #    raise Exception(f'No stats column to filter out data in with {suffixes}: {kwargs_data}. EXITING')
        #interested_stats_col = interested_stats_columns[0]
        
        info = []
        for _, row in df.iterrows():
            sigma = row[sigma_col]
            if is_not_nan(sigma):
                variances = augment_samples(sigma,
                                            lowest_power, highest_power,
                                            n_samples_per_magnitude)
                for var, base, exponent, difference in variances:
                    if is_not_nan(var):
                        sigma_kernel_2 = var - sigma**2
                        if sigma_kernel_2 is None or sigma_kernel_2 < 0:
                            continue
                        sigma_kernel = np.sqrt(sigma_kernel_2)
                        if sigma == 0:
                            sigma_exponent = 0
                            sigma_base = 0
                        else:
                            sigma_exponent = int(np.floor(np.log10(abs(sigma))))
                            sigma_base = int(np.floor(sigma / (10.0 ** sigma_exponent)))

                        info.append({'name' : row[name_col],
                                        'base': base, 'exponent': exponent, 'difference' : difference,
                                        'variance' : var, 'combined_sigma':np.sqrt(var),
                                        'sigma': sigma, 'noisy_sigma': sigma_kernel,
                                        'location':row[location_col], 'dataset' : row[dataset],
                                        'abs_mean':row[abs_mean_col], 'org_abs_mean':row[org_abs_mean_col],
                                        'stats_col':row[interested_stats_col],
                                        'exp_time':row[exposure_col], 'sigma_base':sigma_base,
                                        'sigma_exponent':sigma_exponent})
                    else:
                        continue
            else:
                continue   
        info = pd.DataFrame(info)
        #ranges = find_distribution(df, sigma_col)
        #ranges = find_distribution_only_exp(df, sigma_col)
        info = info.dropna(subset=['sigma', 'noisy_sigma', 'combined_sigma', 'variance',
        'location', 'name', 'abs_mean', 'org_abs_mean', 'stats_col'])
        info = info[(info['variance'] != 0) | (info['sigma'] != 0) | (info['noisy_sigma'] != 0) |
                    (info['abs_mean'] != 0) | (info['org_abs_mean'] != 0) | (info['stats_col'] != 0) |
                    (info['exp_time'] != 0)]
        info['variance'] = info['variance'].abs()
        info['sigma'] = info['sigma'].abs()
        info['noisy_sigma'] = info['noisy_sigma'].abs()
        info['abs_mean'] = info['abs_mean'].abs()
        info['org_abs_mean'] = info['org_abs_mean'].abs()
        info['stats_col'] = info['stats_col'].abs()
        info = info[info['combined_sigma'] > info['sigma']]
        info['conf_index'] = (info['combined_sigma'] - info['sigma']) / (info['combined_sigma'] + info['sigma'])
        info['NSR_org'] = info['abs_mean'].mean()/info['sigma']
        info['NSR_noisy'] = info['abs_mean'].mean()/info['combined_sigma']
        info['NSR_ratio'] = info['NSR_org']/info['NSR_noisy']
        info = info[info['NSR_ratio'] <= ratio]
        info = info[(info['conf_index'] <= conf_up) & (info['conf_index'] >= conf_low)]
        #info['range_valid'] = info.apply(lambda row: apply_ranges(ranges, row), axis=1)
        #info['range_valid'] = info.apply(lambda row: apply_ranges_only_exponent(ranges, row), axis=1)
        #info = info[info['range_valid']]
        info.to_csv('info.csv',index=False)
        #part1 = info[info['exp_time'] >= 10*lowest_exposure]
        #part2 = info[(info['exp_time'] < 10*lowest_exposure) & (info['exp_time'] == 0)]  

        #info = info[info['noisy_sigma'] <= info['org_abs_mean']*abs_percent/100]

        x = min(samples if training else val_samples, len(info))  
        if training:
            info = info[info['location'].apply(lambda x: 'training' in str(x) if pd.notnull(x) else False)]
        else:
            info = info[info['location'].apply(lambda x: 'eval' in str(x) if pd.notnull(x) else False)]
        
        sampled = filtering_df(info, x, delta, 'dataset', 'abs_mean', 'base', 'difference', 'NSR_ratio')
        print(samples if training else val_samples, len(info), x, len(sampled))
        #sampled.to_csv('sampled.csv', index=False)
        #sorted_df = sampled.sort_values(by=['dataset', 'abs_mean', 'base', 'exponent'], ascending=[True, False, False, True])
        #sorted_df[['dataset', 'abs_mean', 'base', 'exponent', 'combined_sigma', 'sigma']].to_csv('sampled_sorted.csv', index=False)
        if training:
            info = sampled
            MEM_CACHED = info
            info.to_csv('./sampled_training.csv', index=False)
            n = int(len(info)*0.25)
            sampled = info.sample(n=n).reset_index(drop=True)
        else:
            info = sampled
            MEM_CACHED_EVAL = info
            info.to_csv('./sampled_eval.csv', index=False)
            n = int(len(info)*0.1)
            sampled = info.sample(n=n).reset_index(drop=True)
    else:
        if training:
            info = MEM_CACHED
            #info.to_csv('./sampled_training.csv', index=False)
            n = int(len(info)*0.25)
            sampled = info.sample(n=n).reset_index(drop=True)
        else:
            info = MEM_CACHED_EVAL
            #info.to_csv('./sampled_eval.csv', index=False)
            n = int(len(info)*0.1)
            sampled = info.sample(n=n).reset_index(drop=True)
    yield from prepare_data_2(sampled, training, data_folder, scaling)

def data_augment_5(images, kwargs_data, scaling=None):
    """
    This is the main augmentation function used. 
    Augments data samples based on the provided metadata and other augmentation parameters.
    
    This function reads metadata from a CSV file and processes each row to apply data augmentation 
    techniques to the images. For each valid row, it computes a range of synthetic variances based 
    on the provided noise characteristics, generates augmented data, and prepares it for further 
    processing. The function yields augmented data samples in batches, which can be used for 
    training or evaluation purposes.

    Args:
        images (list): A list of image file paths to be processed. These are used for sampling.
        kwargs_data (dict): A dictionary containing various parameters necessary for the augmentation 
                            process, including metadata filepath, column names, sample parameters, 
                            and thresholds for processing. The keys are:
                            - 'metadata_filepath' (str): Path to the CSV file containing metadata.
                            - 'exposure_col' (str): The name of the column containing exposure values.
                            - 'name_col' (str): The name of the column containing image file names.
                            - 'sigma_col' (str): The name of the column containing sigma values.
                            - 'location_col' (str): The name of the column indicating location (training/eval).
                            - 'abs_mean_col' (str): The column containing absolute mean values.
                            - 'org_abs_mean_col' (str): The original absolute mean column for the uncropped images.
                            - 'dataset' (str): The dataset identifier.
                            - 'samples' (int): Number of samples to generate.
                            - 'low' (float): Lower threshold for exposure values.
                            - 'high' (float): Upper threshold for exposure values.
                            - 'training' (bool): Whether to filter for training data (True) or evaluation (False).
                            - 'lowest_power' (int): The lowest power for the variance augmentation.
                            - 'highest_power' (int): The highest power for the variance augmentation.
                            - 'n_samples_per_magnitude' (int): Number of samples to generate per variance magnitude.
                            - 'delta' (float): Threshold for filtering data based on absolute mean values.
                            - 'conf_low' (int): Lower threshold for confidence index filtering.
                            - 'conf_up' (int): Upper threshold for confidence index filtering.
                            - 'data_folder' (str): Path to the data folder containing the images.

    Yields:
        tuple: A tuple containing augmented data:
            - Noisy image data (numpy array of shape (height, width, 1)).
            - Clean image data (numpy array of shape (height, width, 1)).
            - Image file name (string, e.g., "image_name_sigma_value.fits").
    
    Raises:
        Exception: If any required parameter in `kwargs_data` is missing or invalid, or if insufficient 
                   samples are available for the augmentation process.
    
    Example:
        # Example usage
        data_generator = data_augment_3(images=["image1.fits", "image2.fits"], kwargs_data=kwargs)
        for noisy_image, clean_image, filename in data_generator:
            # Use the augmented data in training or evaluation
            pass
    """
        
    metadata_filepath = kwargs_data.get('metadata_filepath', None)
    exposure_col = kwargs_data.get('exposure_col', None)
    name_col = kwargs_data.get("name_col", None)
    sigma_col = kwargs_data.get("sigma_col", None)
    location_col = kwargs_data.get("location_col", None)
    abs_mean_col = kwargs_data.get("abs_mean_col", None)
    org_abs_mean_col = kwargs_data.get("org_abs_mean_col", None)
    dataset = kwargs_data.get("dataset", None)
    samples = kwargs_data.get('samples', 5000)
    val_samples = kwargs_data.get('val_samples', 5000)
    low = kwargs_data.get("low", 90)
    high = kwargs_data.get('high', 15000)
    training = kwargs_data.get('training', True)
    lowest_power = kwargs_data.get('lowest_power', -4)
    highest_power = kwargs_data.get('highest_power', 2)
    data_folder = kwargs_data.get('data_folder', None)
    n_samples_per_magnitude = kwargs_data.get('n_samples_per_magnitude', 2) 
    delta = kwargs_data.get('delta', 0.25) 
    conf_low = kwargs_data.get('conf_low', 2)
    conf_up = kwargs_data.get('conf_up', 2)
    times = kwargs_data.get('times', 2)

    columns_to_check = [
        metadata_filepath,
        exposure_col,
        name_col,
        sigma_col,
        location_col,
        abs_mean_col,
        data_folder,
        dataset,
        org_abs_mean_col
    ]
    if any(col is None for col in columns_to_check):
        raise Exception(f'One of the important parameters in the data arguments is None: {kwargs_data}. EXITING')
    conf_up = set_threshold(conf_up)
    conf_low = set_threshold(conf_low, start=True)

    global MEM_CACHED, MEM_CACHED_EVAL
    if MEM_CACHED is not None:
        df = MEM_CACHED
        first_training = False
        first_eval = False
    else:
        df = pd.read_csv(metadata_filepath)
        df = df[[sigma_col, exposure_col, name_col, location_col, abs_mean_col, org_abs_mean_col, dataset]]
        df = df[(df[exposure_col] >= times*low) & (df[exposure_col] <= high)]
        MEM_CACHED = df
        first_training = True
        first_eval = False
    if MEM_CACHED_EVAL is None and not training: 
        first_eval = True
        MEM_CACHED_EVAL = df

    info = []
    for _, row in df.iterrows():
        sigma = row[sigma_col]
        if is_not_nan(sigma):
            variances = augment_samples(sigma,
                                        lowest_power, highest_power,
                                        n_samples_per_magnitude)
            for var, base, exponent, difference in variances:
                if is_not_nan(var):
                    sigma_kernel_2 = var - sigma**2
                    if sigma_kernel_2 is None or sigma_kernel_2 < 0:
                        continue
                    sigma_kernel = np.sqrt(sigma_kernel_2)
                    if sigma == 0:
                        sigma_exponent = 0
                        sigma_base = 0
                    else:
                        sigma_exponent = int(np.floor(np.log10(abs(sigma))))
                        sigma_base = int(np.floor(sigma / (10.0 ** sigma_exponent)))

                    info.append({'name' : row[name_col],
                                    'base': base, 'exponent': exponent, 'difference' : difference,
                                    'variance' : var, 'combined_sigma':np.sqrt(var),
                                    'sigma': sigma, 'noisy_sigma': sigma_kernel,
                                    'location':row[location_col], 'dataset' : row[dataset],
                                    'abs_mean':row[abs_mean_col], 'org_abs_mean':row[org_abs_mean_col],
                                    'exp_time':row[exposure_col], 'sigma_base':sigma_base,
                                    'sigma_exponent':sigma_exponent})
                else:
                    continue
        else:
            continue   

    info = pd.DataFrame(info)
    info = info.dropna(subset=['sigma', 'noisy_sigma', 'combined_sigma', 'variance',
    'location', 'name', 'abs_mean', 'org_abs_mean'])
    info = info[(info['variance'] != 0) | (info['sigma'] != 0) | (info['noisy_sigma'] != 0) |
                (info['abs_mean'] != 0) | (info['org_abs_mean'] != 0)|
                (info['exp_time'] != 0)]
    info['variance'] = info['variance'].abs()
    info['sigma'] = info['sigma'].abs()
    info['noisy_sigma'] = info['noisy_sigma'].abs()
    info['abs_mean'] = info['abs_mean'].abs()
    info['org_abs_mean'] = info['org_abs_mean'].abs()
    info = info[info['combined_sigma'] > info['sigma']]
    info['conf_index'] = (info['combined_sigma'] - info['sigma']) / (info['combined_sigma'] + info['sigma'])
    info = info[(info['conf_index'] <= conf_up) & (info['conf_index'] >= conf_low)]
    if first_training  or first_eval:
        info.to_csv('info.csv',index=False)

    x = min(samples if training else val_samples, len(info))  
    if training:
        info = info[info['location'].apply(lambda x: 'training' in str(x) if pd.notnull(x) else False)]
    else:
        info = info[info['location'].apply(lambda x: 'eval' in str(x) if pd.notnull(x) else False)]
    
    sampled = filtering_df_5(info, x, delta, 'dataset', 'abs_mean', 'base', 'difference')
    if first_training or first_eval:
        print(samples if training else val_samples, len(info), x, len(sampled))

    if training:
        if first_training:
            sampled = sampled.sample(frac=1).reset_index(drop=True)
            sampled.to_csv('./sampled_training.csv', index=False)
    else:
        if first_eval:
            sampled = sampled.sample(frac=1).reset_index(drop=True)
            sampled.to_csv('./sampled_eval.csv', index=False)
    yield from prepare_data_2(sampled, training, data_folder, scaling)

def create_ratios(ratio_initial, ratio_count, ratio_growth):    
    ratios = []
    i = ratio_initial
    for _ in range(ratio_count):
        ratios.append(i)
        i = int(np.ceil(i * ratio_growth))
    return ratios

def turn_exp_to_noisy_sigma(exp, sigma, ratio):
    new_exp = exp / ratio
    new_sigma = np.sqrt(exp / new_exp) * sigma
    return new_sigma, new_exp

def create_noisy_image(image, old_exp, gamma):
    new_exp = old_exp / gamma
    signal = np.maximum(image, 1e-6) 
    signal_scaled = signal * new_exp
    signal_noisy = np.random.poisson(signal_scaled) / new_exp
    noisy_image = signal_noisy
    return np.maximum(noisy_image, 1e-6)

def create_noisy_image_gaussian(image, gamma, median_bkg, std_bkg):
    signal = image-median_bkg
    new_sigma = np.sqrt(gamma)*std_bkg
    noisy_image = signal + np.random.normal(loc=median_bkg, scale=new_sigma, size=image.shape)
    return np.maximum(noisy_image, 1e-6)

def data_augment_last(images, kwargs_data, scaling=None):
    data_folder = kwargs_data.get('data_folder', None)
    name_col = kwargs_data.get("name_col", None)
    training = kwargs_data.get('training', True)
    metadata_filepath = kwargs_data.get('metadata_filepath', None)
    dataset = kwargs_data.get("dataset", None)
    sigma_col = kwargs_data.get("sigma_col", None)
    samples = kwargs_data.get('samples', 5000)
    val_samples = kwargs_data.get('val_samples', 5000)
    times = kwargs_data.get('times', 2)
    low = kwargs_data.get("low", 90)
    high = kwargs_data.get('high', 15000)
    exposure_col = kwargs_data.get('exposure_col', None)
    location_col = kwargs_data.get("location_col", None)
    abs_mean_col = kwargs_data.get("abs_mean_col", None)
    org_abs_mean_col = kwargs_data.get("org_abs_mean_col", None)
    ratio_initial = kwargs_data.get('ratio_initial', 2)
    ratio_count = kwargs_data.get('ratio_count', 10)
    ratio_growth = kwargs_data.get('ratio_growth', 1.5)
    median_col = kwargs_data.get('median_col', None)
    ratios = create_ratios(ratio_initial, ratio_count, ratio_growth)

    columns_to_check = [
        metadata_filepath,
        exposure_col,
        sigma_col,
        location_col,
        abs_mean_col,
        data_folder,
        dataset,
        org_abs_mean_col,
        name_col,
        median_col
    ]
    if any(col is None for col in columns_to_check):
        raise Exception(f'One of the important parameters in the data arguments is None: {kwargs_data}. EXITING')

    global MEM_CACHED, MEM_CACHED_EVAL
    if MEM_CACHED is not None:
        df = MEM_CACHED
        first_training = False
        first_eval = False
    else:
        df = pd.read_csv(metadata_filepath)
        df = df[[sigma_col, exposure_col, name_col, location_col, abs_mean_col, org_abs_mean_col, dataset, median_col]]
        df = df[(df[exposure_col] >= times*low) & (df[exposure_col] <= high)]
        MEM_CACHED = df
        first_training = True
        first_eval = False
    if MEM_CACHED_EVAL is None and not training: 
        first_eval = True
        MEM_CACHED_EVAL = df

    info = []
    for _, row in df.iterrows():
        sigma = row[sigma_col]
        exp = row[exposure_col]
        if is_not_nan(sigma):
            for ratio in ratios:
                
                new_sigma, new_exp = turn_exp_to_noisy_sigma(exp, sigma, ratio)
                noisy_sigma = np.sqrt(new_sigma**2 - sigma**2) if new_sigma**2 > sigma**2 else np.nan
                info.append({
                    'dataset': row[dataset],
                    'name' : row[name_col],
                    'patch' : os.path.basename(row[location_col]),
                    'location': row[location_col],
                    'exp_time': exp,
                    'new_exp_time': new_exp,
                    'combined_sigma': new_sigma,
                    'noisy_sigma': noisy_sigma,
                    'sigma': sigma,
                    'exp_ratio' : round(row[exposure_col]/new_exp, 2),
                    'abs_mean':row[abs_mean_col], 
                    'org_abs_mean':row[org_abs_mean_col],
                    'median' : row[median_col]
                })

    info = pd.DataFrame(info)
    info = info.dropna()
    info = info[info['median'] > 0] 
    info = info[info['combined_sigma'] > info['sigma']]
    info = info[info['exp_time'] > info['new_exp_time']]
    info = info[info['new_exp_time'] >= low]
    info['noisy_sigma'] = info['noisy_sigma'].abs()
    info['sigma'] = info['sigma'].abs()
    info['combined_sigma'] = info['combined_sigma'].abs()
    info['abs_mean'] = info['abs_mean'].abs()
    info['org_abs_mean'] = info['org_abs_mean'].abs()
    info['exp_time'] = info['exp_time'].abs()
    info['new_exp_time'] = info['new_exp_time'].abs()
    info['SNR'] = info['org_abs_mean']/info['sigma']
    info['SNR_noisy'] = info['org_abs_mean']/info['combined_sigma']
    info = info.sample(frac=1).reset_index(drop=True)

    if first_training or first_eval:
        info.to_csv('info.csv',index=False)

    x = min(samples if training else val_samples, len(info))  
    if training:
        info = info[info['location'].apply(lambda x: 'training' in str(x) if pd.notnull(x) else False)]
    else:
        info = info[info['location'].apply(lambda x: 'eval' in str(x) if pd.notnull(x) else False)]
    
    sampled = info.sample(frac=1)

    if first_training or first_eval:
        print(samples if training else val_samples, len(info), x, len(sampled))

    if training:
        if first_training:
            sampled = sampled.reset_index(drop=True)
            sampled.to_csv('./sampled_training.csv', index=False)
    else:
        if first_eval:
            sampled = sampled.reset_index(drop=True)
            sampled.to_csv('./sampled_eval.csv', index=False)
    sampled = sampled.sample(frac=1)
    yield from prepare_data_3(sampled, training, data_folder, scaling)
    
########################BASED ON SNR AND SIGMA ##################################
#################################################################################
def calculate_noisy_sigma(sigma, t1, t2):
    if t2 != 0:
        coeff = t1/t2 - 1
        if coeff > 0:
            return sigma*np.sqrt(coeff)
    return 0

def augment_samples_2(sigma, start, end, n_samples_per_magnitude=3):
    sigmas = []
    if is_not_nan(sigma) and sigma != 0.0:
            magnitudes = np.ceil(np.log10(end))
            numbers = np.linspace(start, end, int(magnitudes*n_samples_per_magnitude+1))[0:-1]
            for number in numbers:
                sigmas.append((calculate_noisy_sigma(sigma, end, number), 0, 0, number))
                #sigmas.append((calculate_noisy_sigma(sigma, end, number), 0, 0, number))
    return sigmas

def filtering_df_2(data, n_samples, delta, id_, abs_, noise_ratio):
    if n_samples >= len(data):
        return data

    data['valid'] = data.apply(lambda row: is_relevant_crop(row, delta=delta), axis=1)
    data = data[data['valid']]

    if n_samples >= len(data):
        return data

    data.loc[:, 'temp_index'] = np.arange(0, len(data))
    dfs = []
    already_in = 0
    while True:
        n_samples -= already_in
        sorted_df = data.sort_values(by=[id_, abs_, noise_ratio], ascending=[True, False, False])
        #sorted_df[[id_, abs_, base, exponent, 'combined_sigma', 'sigma']].to_csv('sorted.csv', index=False)

        result = sorted_df.groupby(id_).apply(
            lambda group: group.drop_duplicates(subset=abs_, keep='first')
        )
        if len(result) >= n_samples:
            dfs.append(result.sort_values(by=[abs_], ascending=False, ignore_index=True).iloc[:n_samples])
            break
        else:
            dfs.append(result)
            data = data[~data['temp_index'].isin(result['temp_index'])]
            already_in += len(result)
            if data.empty:
                break
    return pd.concat(dfs, ignore_index=True)

def data_augment_4(images, kwargs_data, scaling=None):
    metadata_filepath = kwargs_data.get('metadata_filepath')
    exposure_col = kwargs_data.get('exposure_col')
    name_col = kwargs_data.get("name_col")
    sigma_col = kwargs_data.get("sigma_col")
    location_col = kwargs_data.get("location_col")
    org_abs_mean_col = kwargs_data.get("org_abs_mean_col")
    abs_mean_col = kwargs_data.get("abs_mean_col")
    dataset = kwargs_data.get("dataset")
    samples = kwargs_data.get('samples', 16000)
    val_samples = kwargs_data.get('val_samples', 4000)
    low = kwargs_data.get("low", 90)
    high = kwargs_data.get('high', 15000)
    training = kwargs_data.get('training', True)
    powers = kwargs_data.get('powers', 3)
    data_folder = kwargs_data.get('data_folder')
    n_samples_per_magnitude = kwargs_data.get('n_samples_per_magnitude', 2) 
    delta = kwargs_data.get('delta', 0.25) 
    conf_low = kwargs_data.get('conf_low', 2)
    conf_up = kwargs_data.get('conf_up', 2)
    times = kwargs_data.get('times', 2)
    min_ratio = kwargs_data.get('min_ratio', 0)
    ratio = kwargs_data.get('ratio', 1000)

    global MEM_CACHED, MEM_CACHED_EVAL
    condition = MEM_CACHED is None if training else MEM_CACHED_EVAL is None
    if condition:
        columns_to_check = [
            metadata_filepath, exposure_col, name_col, sigma_col,
            location_col, abs_mean_col, org_abs_mean_col, data_folder, dataset
        ]

        if any(col is None for col in columns_to_check):
            raise Exception(f'One of the important parameters in the data arguments is None: {kwargs_data}. EXITING')

        conf_up = set_threshold(conf_up)
        conf_low = set_threshold(conf_low, start=True)

        df = pd.read_csv(metadata_filepath)
        df = df[(df[exposure_col] >= times * low) & (df[exposure_col] <= high)]

        info = []
        for _, row in df.iterrows():
            sigma = row[sigma_col]
            if is_not_nan(sigma):
                sigmas = augment_samples_2(sigma, low, row[exposure_col], n_samples_per_magnitude)
                for noisy_sigma, difference, base, time_ratio in sigmas:
                    combined_sigma = np.sqrt(sigma**2 + noisy_sigma**2)
                    info.append({
                        'name': row[name_col],
                        'combined_sigma': combined_sigma,
                        'noisy_sigma': noisy_sigma,
                        'sigma': sigma,
                        'location': row[location_col],
                        'dataset': row[dataset],
                        'abs_mean': row[abs_mean_col],
                        'org_abs_mean': row[org_abs_mean_col],
                        'exp_time': row[exposure_col],
                        'org_signal_to_noise': row[abs_mean_col] / sigma,
                        'aug_signal_to_noise': row[abs_mean_col] / combined_sigma,
                        #'difference': difference,
                        #'base': base,
                        #'new_exp_time' : row[exposure_col]*time_ratio
                        'new_exp_time' : time_ratio
                    })

        info = pd.DataFrame(info).dropna()
        info = info[(info['sigma'] != 0) | (info['noisy_sigma'] != 0) |
                     (info['combined_sigma'] != 0) | (info['abs_mean'] != 0) |
                     (info['org_abs_mean'] != 0) | (info['aug_signal_to_noise'] != 0) |
                     (info['org_signal_to_noise'] != 0) | (info['exp_time'] != 0)]
        abs_columns = ['sigma', 'noisy_sigma', 'combined_sigma', 'abs_mean', 'org_abs_mean',
                        'org_signal_to_noise', 'aug_signal_to_noise']
        info[abs_columns] = info[abs_columns].abs()

        info = info[info['new_exp_time'] >= low]
        info = info[info['combined_sigma'] > info['sigma']]
        info['conf_index'] = (info['combined_sigma'] - info['sigma']) / (info['combined_sigma'] + info['sigma'])
        info = info[(info['conf_index'] <= conf_up) & (info['conf_index'] >= conf_low)]
        info.to_csv('./info.csv', index=False)
        info = info[info['aug_signal_to_noise'] * ratio >= info['org_signal_to_noise']]
        #info = info[info['aug_signal_to_noise'] >= min_ratio]

        if not len(info):
            raise Exception('No adequate sampling is possible')

        x = min(samples if training else val_samples, len(info))    
        if training:
            info = info[info['location'].str.contains('training', na=False)]
        else:
            info = info[info['location'].str.contains('eval', na=False)]

        #sampled = filtering_df(info, x, delta, 'dataset', 'abs_mean', 'base', 'difference')
        sampled = filtering_df_2(info, x, delta, 'dataset', 'abs_mean', 'aug_signal_to_noise')
        print(samples if training else val_samples, len(info), x, len(sampled))
        if training:
            info = sampled
            MEM_CACHED = info
            info.to_csv('./sampled_training.csv', index=False)
            n = int(len(info)*0.25)
            sampled = info.sample(n=n).reset_index(drop=True)
        else:
            info = sampled
            MEM_CACHED_EVAL = info
            info.to_csv('./sampled_eval.csv', index=False)
            n = int(len(info)*0.1)
            sampled = info.sample(n=n).reset_index(drop=True)
    else:
        if training:
            info = MEM_CACHED
            #info.to_csv('./sampled_training.csv', index=False)
            n = int(len(info)*0.25)
            sampled = info.sample(n=n).reset_index(drop=True)
        else:
            info = MEM_CACHED_EVAL
            #info.to_csv('./sampled_eval.csv', index=False)
            n = int(len(info)*0.1)
            sampled = info.sample(n=n).reset_index(drop=True)
    yield from prepare_data_2(sampled, training, data_folder, scaling)

def create_tf_dataset(images, sample_generator, generator_kwargs, batch_size=32, scaling=None, rotation_range=10):
    """
    Creates a TensorFlow Dataset with small random rotations, flips, and 90° rotations.

    Args:
        images (list): List of image file paths.
        sample_generator (callable): Generator function yielding (noisy image, clean image, file path).
        generator_kwargs (dict): Keyword arguments passed to the sample generator.
        batch_size (int, optional): Batch size. Default is 32.
        scaling (str, optional): Scaling method. Default is None.
        rotation_range (int, optional): Max rotation in degrees (e.g., 10° results in [-10°, 10°]).

    Returns:
        tf.data.Dataset: Augmented TensorFlow dataset.
    """
    shape = 1
    if scaling in ['log_min_max']:
        shape = 7
    elif scaling in ['z_scale', 'min_max']:
        shape = 5

    def generator():
        yield from sample_generator(images=images, kwargs_data=generator_kwargs, scaling=scaling)

    ps = generator_kwargs.get('ps', 256)
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(ps, ps, 1), dtype=tf.float32),  # x_train (noisy)
            tf.TensorSpec(shape=(ps, ps, 1), dtype=tf.float32),  # y_train (clean)
            tf.TensorSpec(shape=(shape), dtype=tf.string)  # file paths
        )
    )

    def apply_augmentations(x, y, z):
        """Applies small random rotations, flips, and 90° rotations."""
        # Small random rotation (-rotation_range° to +rotation_range°)
        #angle = tf.random.uniform([], -rotation_range, rotation_range) * np.pi / 180
        #x = tfa.image.rotate(x, angle, interpolation="BILINEAR")
        #y = tfa.image.rotate(y, angle, interpolation="BILINEAR")

        # Random 90° rotation (0°, 90°, 180°, 270°)
        k = tf.random.uniform([], 0, 4, dtype=tf.int32)  # Random integer 0-3
        x = tf.image.rot90(x, k)
        y = tf.image.rot90(y, k)

        # Random horizontal & vertical flips
        #x = tf.image.random_flip_left_right(x)
        #x = tf.image.random_flip_up_down(x)
        #y = tf.image.random_flip_left_right(y)
        #y = tf.image.random_flip_up_down(y)

        return x, y, z

    dataset = dataset.map(apply_augmentations, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset

############ CHECKPOINTS ##########################################################################
###############################################################################################################
###############################################################################################################

def restore_model(checkpoint_dir, checkpoint_prefix, epoch):
    """
    Restores a model from a checkpoint file.

    This helper function loads a model from a checkpoint file, if it exists, based on the specified 
    epoch. The model is loaded from a `.keras` file located in the provided checkpoint directory. 
    If the checkpoint file is successfully loaded, the model and epoch number are returned.

    Args:
        checkpoint_dir (str): The directory where the checkpoint files are stored.
        checkpoint_prefix (str): The prefix used to name the checkpoint files.
        epoch (int): The epoch number of the checkpoint file to restore.

    Returns:
        tuple: A tuple containing the restored model and epoch number if successful.
               If the checkpoint file does not exist or cannot be loaded, returns None.

    Example:
        # Example usage
        model, epoch = restore_model('checkpoints', 'model_checkpoint', 10)
        if model is not None:
            print(f"Model restored from epoch {epoch}.")
        else:
            print("Failed to restore the model.")
    """
    filename = f'{checkpoint_prefix}_{epoch:04d}.keras'
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    if os.path.exists(checkpoint_path):
        try:
            model = tf.keras.models.load_model(checkpoint_path)
            logging.warning(f'restored the model file: {filename}, epoch:{epoch}')
            return model, epoch
        except Exception as err:
            logging.warning(f"An error occurred while trying to restore the epoch {epoch}: {err}")
    return None

def load_model(checkpoint_dir, start_from_best, start_from_last, start_from_scratch):
    """
    Loads a model from a checkpoint based on the provided flags.

    This function attempts to restore a model from a checkpoint directory. It handles three possible 
    scenarios: starting from the best checkpoint, starting from the last checkpoint, or starting 
    from scratch. If both `start_from_best` and `start_from_last` are True, it raises an error.
    
    The function looks for checkpoint files that start with 'best_' (indicating the best model 
    checkpoints) and 'model_' (indicating general model checkpoints). If a checkpoint is found, 
    it attempts to restore the model from the corresponding file.

    Args:
        checkpoint_dir (str): The directory where the checkpoint files are stored.
        start_from_best (bool): Flag to indicate whether to restore the best model checkpoint.
        start_from_last (bool): Flag to indicate whether to restore the last model checkpoint.
        start_from_scratch (bool): Flag to indicate whether to start training from scratch.

    Returns:
        int: The epoch from which the model is restored. If no checkpoint is found, returns 0 
             (indicating training will start from scratch).

    Raises:
        ValueError: If both `start_from_best` and `start_from_last` are True.

    Example:
        # Example usage
        start_epoch = load_model('checkpoints', True, False, False)
        print(f"Model restored from epoch {start_epoch}.")
    """

    start_epoch = 0
    if start_from_best == start_from_last:
        raise ValueError("Cannot start from both 'best' and 'last' checkpoints simultaneously.")
    if start_from_scratch and os.path.exists(checkpoint_dir):
        #shutil.rmtree(checkpoint_dir)
        return start_epoch
    
    # Get lists of available best epochs and model epochs
    if os.path.exists(checkpoint_dir):
        best_epochs = sorted({int(f.split(".")[0].split("_")[-1]) for f in os.listdir(checkpoint_dir) if f.startswith('best_')})
        epochs = sorted({int(f.split(".")[0].split("_")[-1]) for f in os.listdir(checkpoint_dir) if f.startswith('model_')})
        # Try restoring from the best epochs first
        if best_epochs and start_from_best and not start_from_last:
            for epoch in reversed(best_epochs):
                result = restore_model(checkpoint_dir, 'best_model', epoch)
                if result:
                    return result
            start_from_best = False
            start_from_last = True
            logging.warning('no best checkpoint has been found, changing to the last working checkpoint')

        # If no best epoch was restored, or we are falling back to the last checkpoint
        if epochs and start_from_last and not start_from_best:
            for epoch in reversed(epochs):
                result = restore_model(checkpoint_dir, 'model', epoch)
                if result:
                    return result

    if not start_epoch:
        logging.warning("No valid checkpoint found, starting from scratch.")
    return 0

################TRAIN THE MODEL ################################
def train_network(input_shape, folder, n_epochs, kwargs_data, kwargs_network, data_generator, batch_size=32, network_name='model',
         optimizer=Adam, change_learning_rate=[(0, 1e-4), (2000, 1e-5)], G_loss_fn=tf.keras.losses.MeanAbsoluteError(),
         start_from_scratch=False, start_from_best=False, start_from_last=True,
         save_freq=500, eval_percent=20, period_save=30, scaling=None):
    """
    Trains a neural network model using the provided dataset and configuration.

    This function sets up and trains a neural network model, including data preparation, model restoration
    (from best or last checkpoint), and saving the model during training. The training process includes 
    periodic evaluations and saving of model checkpoints based on the specified parameters.

    Args:
        input_shape (tuple): Shape of the input data (e.g., (256, 256, 3)).
        folder (str): Directory where the model and checkpoints will be saved.
        n_epochs (int): Number of epochs to train the model.
        kwargs_data (dict): Dictionary containing data-related parameters, including paths to training and evaluation data.
        kwargs_network (dict): Dictionary containing the network-specific parameters.
        data_generator (function): Function to generate batches of data.
        batch_size (int, optional): Batch size for training. Defaults to 32.
        network_name (str, optional): Name of the network. Defaults to 'model'.
        optimizer (tf.keras.optimizers.Optimizer, optional): Optimizer for training. Defaults to Adam.
        change_learning_rate (list, optional): List of tuples specifying when and what learning rate to use. Defaults to [(0, 1e-4), (2000, 1e-5)].
        G_loss_fn (callable, optional): Loss function used for training. Defaults to tf.keras.losses.MeanAbsoluteError().
        start_from_scratch (bool, optional): Flag to indicate whether to start from scratch (no checkpoint). Defaults to False.
        start_from_best (bool, optional): Flag to indicate whether to start from the best checkpoint. Defaults to False.
        start_from_last (bool, optional): Flag to indicate whether to start from the last checkpoint. Defaults to True.
        save_freq (int, optional): Frequency (in steps) to save model checkpoints. Defaults to 500.
        eval_percent (int, optional): Percentage of the dataset to use for evaluation. Defaults to 20.
        period_save (int, optional): Period (in epochs) to save model checkpoints. Defaults to 30.

    Raises:
        FileNotFoundError: If the paths for training or evaluation data do not exist.
        ValueError: If 'G_loss_fn' is not callable.

    Returns:
        None: The function trains the model and saves the training history.

    Example:
        # Example usage
        train_network(
            input_shape=(256, 256, 1),
            folder='path/to/save/models',
            n_epochs=100,
            kwargs_data={'training_path': 'train/', 'eval_path': 'eval/'},
            kwargs_network={'num_layers': 3, 'filters': 64},
            data_generator=my_data_generator
        )
    """

    # Ensures that the dateset exists and checks
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    os.chdir(folder)
    if not callable(G_loss_fn):
        raise ValueError("The 'G_loss_fn' must be callable.")
    
    #Reading the imge filepaths from files residing in the respective folders

    training_path = kwargs_data.get("training_path", None)
    eval_path = kwargs_data.get("eval_path", None)
    if training_path is not None and os.path.exists(training_path):
        train_data = [f'{training_path}/{f}' for f in os.listdir(training_path) if f.endswith('.fits')]
    else:
        raise FileNotFoundError(f'No file at: {training_path}')
    if eval_path is not None and os.path.exists(eval_path):
        eval_data = [f'{eval_path}/{f}' for f in os.listdir(eval_path) if f.endswith('.fits')]
    else:
        raise FileNotFoundError(f'No file at: {eval_path}')
    
    #Creation of folders
    try:
        results_folder = f'./{network_name}'
        if not os.path.exists(results_folder):
            os.makedirs(results_folder, exist_ok=True)
        os.chdir(results_folder)
    except Exception as err:
        logging.warning(f'an error occurred while finding the output path: {err}')
        return

    # Model definition
    start_epoch = 0
    model = network(input_shape, **kwargs_network)
    optimizer = optimizer()
    train_dataset = create_tf_dataset(train_data,
                                    data_generator,
                                    generator_kwargs=kwargs_data,
                                    batch_size=batch_size, scaling=scaling)

    # Restoring prior epochs
    checkpoint_dir = './checkpoints'
    result = load_model(checkpoint_dir, start_from_best, start_from_last, start_from_scratch)
    if result:
        model, start_epoch = result
    model.compile(optimizer=optimizer, loss=G_loss_fn)
    
    if 'training' in kwargs_data:
        kwargs_val = dict(kwargs_data)
        kwargs_val['training'] = False
    else:
        kwargs_val = kwargs_data

    callback = Callback(
        dataset= train_dataset,
        dataset_size= sum(1 for _ in train_dataset) * batch_size,
        epoch=n_epochs,
        start_epoch = start_epoch, 
        eval_percent=eval_percent,  
        period_save=period_save,  
        save_freq=save_freq,  
        sample_generator=data_generator,
        kwargs_validation=kwargs_val,
        validation_images = eval_data,
        optimizer=optimizer,
        change_learning_rate=change_learning_rate,
        batch_size = batch_size,
        scaling = scaling
    )
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    #Train the model
    history = model.fit(train_dataset.map(lambda x, y, filepath: (x, y)), epochs=n_epochs-start_epoch, callbacks=[callback])   
    try:
        with open('./training_history.json', 'w') as f:
            json.dump(history.history, f)
    except Exception as err:
        logging.warning(f'an error occurred while saving the history: {err}')

######################lOSS FUNCTIONS#############################
def scale_invariant_mae(y_true, y_pred):
    # Get the range of the ground truth image (y_true) for each image in the batch
    range_true = tf.reduce_max(y_true, axis=(1, 2, 3), keepdims=True) - tf.reduce_min(y_true, axis=(1, 2, 3), keepdims=True) + 1e-10
    # Compute the Mean Absolute Error (MAE) across the height, width, and channels
    mae = tf.reduce_mean(tf.abs(y_true - y_pred), axis=(1, 2, 3))
    # Normalize by the range of the ground truth
    return tf.reduce_mean(mae / range_true)

def log_cosh_loss(y_true, y_pred):
    # Get the range of the ground truth image (y_true) for each image in the batch
    range_true = tf.reduce_max(y_true, axis=(1, 2, 3), keepdims=True) - tf.reduce_min(y_true, axis=(1, 2, 3), keepdims=True) + 1e-10
    # Compute the Log-Cosh loss
    log_cosh = tf.reduce_mean(tf.math.log(tf.cosh(y_pred - y_true)), axis=(1, 2, 3))
    # Normalize by the range of the ground truth
    return tf.reduce_mean(log_cosh / range_true)

def main():
    
    data_dir = 'our_data'
    data_dir = 'nsigma2_fprint_5_npixels_5'
    #data_dir = 'gal_data'
    if len(sys.argv) > 3:
        data_dir = sys.argv[3]
    kwargs_data = {
        'type_of_image':'SCI',
        'ps' : 256, 
        'steps' : 100,
        'dk':7, 
        'ron':3,
        'save':False,
        'sv_name':None,
        'path':None,
        'low' : 60,
        'high' : 10000,
        'start' : 2,
        'stop' : 6,
        'training_path' :f'{os.path.dirname(__file__)}/original_dataset/training',
        'eval_path' :f'{os.path.dirname(__file__)}/original_dataset/eval'
    }
    kwargs_data_2 = {
        'metadata_filepath' : f'{os.path.dirname(__file__)}/{data_dir}/cropped_images_stats.csv',
        'exposure_col' : 'sci_actual_duration', 
        'name_col' : 'filename',
        'sigma_col' : 'org_std',
        'fit_data_filepath' : f'{os.path.dirname(__file__)}/{data_dir}/fit_info.csv',
        'fit_data_cols' : ['t', 'a', 'dt', 'da'],
        'samples' : 1000,
        'steps' : 100,
        'low' : 90, 
        'high' : 15000,
        'location_col' : 'location',
        'abs_mean_col' : 'abs_mean',
        'training_path' :f'{os.path.dirname(__file__)}/{data_dir}/training',
        'eval_path' :f'{os.path.dirname(__file__)}/{data_dir}/eval',
        'data_folder' : f'{data_dir}'
    }

    kwargs_data_3 = {
        'metadata_filepath' : f'{os.path.dirname(__file__)}/{data_dir}/cropped_images_stats.csv',
        'exposure_col' : 'sci_actual_duration', 
        'name_col' : 'filename',
        #'sigma_col': 'org_std',
        'sigma_col': 'org_light_std',
        'samples': 3000,
        'val_samples' : 32,
        'low' : 200, 
        'high' : 20000,
        'location_col' : 'location',
        #'abs_mean_col': 'abs_mean',
        'abs_mean_col': 'mean',
        #'org_abs_mean_col': 'org_abs_mean',
        'org_abs_mean_col': 'org_mean',
        'training_path' :f'{os.path.dirname(__file__)}/{data_dir}/training',
        'eval_path' :f'{os.path.dirname(__file__)}/{data_dir}/eval',
        'lowest_power' : -4,
        'highest_power' : 4,
        'data_folder' : f'{data_dir}',
        'median_col' : 'org_median',
        'n_samples_per_magnitude' : 3,
        'delta' : 0.50,
        'dataset' : 'sci_data_set_name',
        'cutoff' : 94,
        'suffixes' : ['mean'],
        'n_sigma_stat' : 2,
        'light_col' : 'light_median',
        'abs_percent' : 35,
        'conf_low' : 2,
        'conf_up' : 2,
        'threshold' : 100,
        'lowest_exposure' : 200,
        'times' : 2,
        'training' : True,
        'ratio': 10,
    }

    kwargs_data_4 = {
    'metadata_filepath': f'{os.path.dirname(__file__)}/{data_dir}/cropped_images_stats.csv',
    'training_path' :f'{os.path.dirname(__file__)}/{data_dir}/training',
    'eval_path' :f'{os.path.dirname(__file__)}/{data_dir}/eval',
    'exposure_col': 'sci_actual_duration', 
    'name_col': 'filename',
    'sigma_col': 'org_light_std',
    'location_col': 'location',
    'abs_mean_col': 'mean',
    'org_abs_mean_col': 'org_mean',
    'dataset': 'sci_data_set_name',
    'samples': 16000,
    'val_samples' : 4000,
    'low': 200,
    'high': 20000,
    'training': True,
    'powers': 3,
    'data_folder': f'{data_dir}',
    'n_samples_per_magnitude': 5,
    'delta': 0.25,
    'conf_low': 2,
    'conf_up': 2,
    'times': 2,
    'min_ratio': 0.5,
    'ratio': 10
    }

    kwargs_data_last = {
    'metadata_filepath' : f'{os.path.dirname(__file__)}/{data_dir}/cropped_images_stats.csv',
    'name_col': 'filename',
    'exposure_col' : 'actual_exp_time', 
    'sigma_col': 'org_light_std',
    'samples': 3000,
    'val_samples' : 10,
    'low' : 200, 
    'high' : 20000,
    'location_col' : 'location',
    'abs_mean_col': 'mean',
    'org_abs_mean_col': 'org_mean',
    'median_col' : 'org_light_median',
    'training_path' :f'{os.path.dirname(__file__)}/{data_dir}/training',
    'eval_path' :f'{os.path.dirname(__file__)}/{data_dir}/eval',
    'data_folder' : f'{data_dir}',
    'dataset' : 'sci_data_set_name',
    'lowest_exposure' : 200,
    'times' : 2,
    'training' : True,
    'ratio_initial' : 2,
    'ratio_count' : 10, 
    'ratio_growth' : 1.5
    }
    
    kwargs_network = {
        'depth':5,
        'kernel_size':3,
        'filter_size':2,
        'pooling_size':2,
        'n_of_initial_channels':32, 
        'func':tf.keras.layers.LeakyReLU,
        'batch_normalization':True,
        'exp' : None,
        'exp_time':None,
        'dropout_rate':0.1
    }

    input_shape = (256, 256, 1)
    folder = './models'
    batch_size = 16
    n_epochs = 1000
    optimizer=Adam
    change_learning_rate=[(0, 1e-4), (1000, 3e-5)]
    G_loss_fn=MeanAbsoluteError()
    #G_loss_fn = scale_invariantu_mae
    #G_loss_fn = log_cosh_loss
    start_from_scratch=False
    start_from_best=False
    start_from_last=True
    save_freq=25
    eval_percent=5
    period_save=5

    network_name='model_gpu_5'
    network_name='model_gpu_scaling_min_max_random'
    network_name='z_scale_1e4_random'
    #network_name='model_gpu_no_scaling_3_e5_random'
    network_name='no_scaling_1e4_random'
    #scaling='min_max'
    #scaling='z_scale'
    scaling=None

    if len(sys.argv) > 1:
        network_name = sys.argv[1]
    if len(sys.argv) > 2:
        if sys.argv[2] != 'None':
            scaling = sys.argv[2]

    directory_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(directory_path)

    logging.warning(f"Num GPUs Available: {len(tf.config.experimental.list_physical_devices('GPU'))}")
    train_network(input_shape, folder, n_epochs, kwargs_data_4, kwargs_network,
                  data_generator=data_augment_5, batch_size=batch_size,
                  network_name=network_name,
                  optimizer=optimizer,change_learning_rate=change_learning_rate,
                  G_loss_fn=G_loss_fn, start_from_scratch=start_from_scratch,
                  start_from_best=start_from_best, start_from_last=start_from_last,
                  save_freq=save_freq,
                  eval_percent=eval_percent, period_save=period_save, scaling=scaling
                  )

MEM_CACHED = None
MEM_CACHED_EVAL = None
if __name__ == "__main__":
    main()