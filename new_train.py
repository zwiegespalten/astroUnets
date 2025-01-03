import os, logging, time, shutil, json
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanAbsoluteError
from astropy.io import fits
from network import network

class Callback(tf.keras.callbacks.Callback):
    def __init__(self, dataset, dataset_size, epoch, start_epoch, eval_percent,
                 period_save, save_freq, sample_generator, kwargs_validation,
                validation_images, optimizer, change_learning_rate):
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

    def create_directory(self, path):
        """Helper function to create a directory if it does not exist."""
        try:
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
        except Exception as err:
            logging.warning(f"An error occurred while creating the directory {path}: {err}")

    def on_train_begin(self, logs=None):
        self.train_start = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()
        for change_epoch, lr in self.change_learning_rate:
            if self.current_epoch >= change_epoch:
                self.optimizer.learning_rate.assign(lr)
                logging.warning(f'epoch: {self.current_epoch} changing to lr: {lr}')

    def on_epoch_end(self, epoch, logs=None):
        self.current_epoch += 1
        train_loss = logs.get('loss', 0) if logs else 0
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
                            filepath = filepath.numpy()[0].decode('utf-8') 
                            sv_name = "/" + filepath.split('/')[-1][:-5]
                            save_fits(y_train[:, :, 0], sv_name, result_path)
                            save_fits(prediction, sv_name + "_output", result_path)
                            save_fits(x_train[:, :, 0], sv_name + "_noise", result_path)
                            temp += 1
                    indices.add(index)

        # PERIODIC VALIDATION
        self.validation_start = time.time()
        count = self.dataset_size * max(int(self.eval_percent), 1) // 100
        indices = np.random.randint(0, len(self.validation_images), 
                                     size=min(count, len(self.validation_images)))
        selected_images = np.array(self.validation_images)[indices]
        validation_data = create_tf_dataset(selected_images, self.sample_generator,
                                                 generator_kwargs=self.kwargs_validation)

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
############################################################

def open_fits(filepath, ratio, type_of_image='SCI', low=60, high=10000):
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
    
def save_fits(image, name, path):
    try:
        hdu = fits.PrimaryHDU(image)
        hdul = fits.HDUList([hdu])
        hdul.writeto(path + name + '.fits', overwrite=True)
        return True
    except Exception as err:
        logging.warning(f'an error occurred while saving the image: {err}')
        return
    
#############CROPPING, NOISE ADDITION AND DATA AUGMENTATIONS###################################################
###############################################################################################################
###############################################################################################################
def poisson_noise(img_data, ratio=0.5, ex_time=1, ron=3, dk=7, save=False,
                   sv_name=None, path=None):
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

def data_augment(images, kwargs_data):
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

def black_level(H, W, out, ps=256, steps=100):
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

def create_tf_dataset(images, sample_generator, generator_kwargs, batch_size=32):
    """
    Creates a TensorFlow Dataset from the data_augment generator with optimized parallelization.
    """
    def generator():
        yield from sample_generator(images=images, kwargs_data=generator_kwargs)
    
    ps = generator_kwargs.get('ps', 256)
    dataset = tf.data.Dataset.from_generator(generator,
                                              output_signature=(
                                                  tf.TensorSpec(shape=(ps, ps, 1), dtype=tf.float32),  # x_train
                                                  tf.TensorSpec(shape=(ps, ps, 1), dtype=tf.float32),  # y_train
                                                  tf.TensorSpec(shape=(1), dtype=tf.string)              # filepaths
                                              ))

    dataset = dataset.map(lambda x, y, z: (tf.convert_to_tensor(x), tf.convert_to_tensor(y), tf.convert_to_tensor(z)),
                          num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.shuffle(buffer_size=100) 
    dataset = dataset.batch(batch_size)          
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

############ CHECKPOINTS ##########################################################################
###############################################################################################################
###############################################################################################################

def restore_model(checkpoint_dir, checkpoint_prefix, epoch):
    """Helper function to restore a checkpoint."""
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
    """Load the appropriate checkpoint based on the flags."""
    start_epoch = 0
    if start_from_best == start_from_last:
        raise ValueError("Cannot start from both 'best' and 'last' checkpoints simultaneously.")
    if start_from_scratch and os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
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
         save_freq=500, eval_percent=20, period_save=30):
    
    # Ensures that the dateset exists and checks
    if os.path.exists(folder):
        os.chdir(folder)
    else:
        raise FileNotFoundError(f'No directory: {folder}')
    if not callable(G_loss_fn):
        raise ValueError("The 'G_loss_fn' must be callable.")
    
    #Reading the imge filepaths from files residing in the respective folders
    training_path = f'{os.getcwd()}/training'
    eval_path = f'{os.getcwd()}/eval'
    if os.path.exists(training_path):
        train_data = [f'{training_path}/{f}' for f in os.listdir(training_path) if f.endswith('.fits')]
    else:
        raise FileNotFoundError(f'No file at: {training_path}')
    if os.path.exists(eval_path):
        eval_data = [f'{eval_path}/{f}' for f in os.listdir(eval_path) if f.endswith('.fits')]
    else:
        raise FileNotFoundError(f'No file at: {eval_path}')
    
    #Creation of folders
    try:
        results_folder = os.path.join(os.getcwd(), 'models', network_name)
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
                                    batch_size=batch_size)

    # Restoring prior epochs
    checkpoint_dir = './checkpoints'
    result = load_model(checkpoint_dir, start_from_best, start_from_last, start_from_scratch)
    if result:
        model, start_epoch = result
    model.compile(optimizer=optimizer, loss=G_loss_fn)
    
    callback = Callback(
        dataset= train_dataset,
        dataset_size= sum(1 for _ in train_dataset) * batch_size,
        epoch=n_epochs,
        start_epoch = start_epoch, 
        eval_percent=eval_percent,  
        period_save=period_save,  
        save_freq=save_freq,  
        sample_generator=data_generator,
        kwargs_validation=kwargs_data,
        validation_images = eval_data,
        optimizer=optimizer,
        change_learning_rate=change_learning_rate
    )
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    #Train the model
    history = model.fit(train_dataset.map(lambda x, y, filepath: (x, y)), epochs=n_epochs-start_epoch, callbacks=[callback])   
    try:
        with open('./training_history.json', 'w') as f:
            json.dump(history.history, f)
    except Exception as err:
        logging.warning(f'an error occurred while saving the history: {err}')

def __main__():

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
        'stop' : 6
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
        'exp_time':None
    }

    input_shape = (256, 256, 1)
    folder = './original_dataset'
    batch_size = 4
    n_epochs = 10
    network_name='model'
    optimizer=Adam
    change_learning_rate=[(0, 1e-4), (1, 1e-5)]
    G_loss_fn=MeanAbsoluteError()
    start_from_scratch=False
    start_from_best=False
    start_from_last=True
    save_freq=1
    eval_percent=1
    period_save=1

    directory_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(directory_path)

    logging.warning(f"Num GPUs Available: {len(tf.config.experimental.list_physical_devices('GPU'))}")
    train_network(input_shape, folder, n_epochs, kwargs_data, kwargs_network,
                  data_generator=data_augment, batch_size=batch_size,
                  network_name=network_name,
                  optimizer=optimizer,change_learning_rate=change_learning_rate,
                  G_loss_fn=G_loss_fn, start_from_scratch=start_from_scratch,
                  start_from_best=start_from_best, start_from_last=start_from_last,
                  save_freq=save_freq,
                  eval_percent=eval_percent, period_save=period_save
                  )
__main__()