import os, logging
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy.io import fits
from photutils.segmentation import detect_threshold, detect_sources
from photutils.utils import circular_footprint
from mast import download_images
from new_train import open_fits, save_fits

####### HELPER FUNCTIONS###########################
def create_dir(save_dir):
    """
    Creates a directory if it doesn't already exist.
    This function attempts to create a directory at the specified path. If the directory already exists, it will
    not raise an error. If an error occurs during the directory creation, it logs the error and returns False.

    Args:
        save_dir (str): Path to the directory that needs to be created.

    Returns:
        bool: True if the directory was successfully created (or already exists), False if an error occurred.

    Example:
        # Example usage
        success = create_dir('path/to/directory')
        if success:
            print("Directory created successfully.")
        else:
            print("Failed to create directory.")
    """
    try:
        os.makedirs(save_dir, exist_ok=True)
        return True
    except Exception as err:
        logging.warning(f'an error occurred while creating the directory: {save_dir}, {err}')
        return False
    
def masked_open_fits(filepath):
    """
    Opens a FITS file and extracts the first valid image data array.
    Args:
        filepath (str): Path to the FITS file.

    Returns:
        tuple or None: A tuple (data, filepath) if a valid image is found, otherwise None.
    """
    try:
        with fits.open(filepath) as f:
            for hdu in f:
                if isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU)) and isinstance(hdu.data, np.ndarray):
                    return hdu.data, filepath
            logging.warning(f"No valid image data found in {filepath}")
            return None
    except Exception as e:
        logging.warning(f"Error reading {filepath}: {e}")
        return None
    
def plot_histogram(df, exp_column, filename):
    """
    Plots a histogram for a specified column in a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        exp_column (str): Column name to plot.
        filename (str): Path to save the histogram plot.
        bins (int): Number of bins for the histogram (default is 30).

    Returns:
        None: Saves the plot to the specified filename.
    """
    mean_val = round(df[exp_column].mean(), 2)
    median_val = round(df[exp_column].median(), 2)
    variance_val = round(df[exp_column].var(), 2)
    std_val = round(df[exp_column].std(), 2)

    plt.figure(figsize=(10, 6))
    plt.hist(df[exp_column], bins=30, color='skyblue', edgecolor='black', label=f'N: {len(df)}')
    plt.title(f'Histogram of {exp_column}')
    plt.xlabel(exp_column)
    plt.ylabel('Frequency')
    plt.grid(True)

    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val}')
    plt.axvline(median_val, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_val}')
    plt.axvline(mean_val + std_val, color='orange', linestyle='dashed', linewidth=2, label=f'Std Dev: {std_val}')
    plt.axvline(mean_val - std_val, color='orange', linestyle='dashed', linewidth=2)

    plt.legend()
    plt.savefig(filename)
    plt.close()

################### IMAGE CROPPING AND STATISTICS #########################
def crop_image(image, filename, save_dir, ps=256):
    """
    Crops an image into smaller patches of a specified size and saves the cropped patches.

    Parameters:
        image (numpy.ndarray): The input image to be cropped.
        filename (str): The original filename of the image.
        save_dir (str): The directory where the cropped patches will be saved.
        ps (int, optional): The size of the cropped patches (patch size). Defaults to 256.

    Returns:
        None

    Notes:
        - If the image dimensions are smaller than the patch size, a warning is logged, and no cropping occurs.
        - Cropped patches are saved in the specified directory with filenames in the format:
          `<original_filename>_<row_index>_<col_index>.fits`.

    Raises:
        - Logs a warning if the input image is not a `numpy.ndarray`.
        - Logs a warning if the save directory cannot be created or the image is too small to crop.
    """
    if not isinstance(image, np.ndarray):
        logging.warning(f"the image is not an 'numpy.ndarray' but {type(image)}")
        return
    if not os.path.exists(save_dir):
        if not create_dir(save_dir):
            return
        
    H, W = image.shape
    if H < ps or W < ps:
        logging.warning(f"the size of the image {(H, W)} is smaller than {ps}")
        return
    
    n_H = int(np.floor(H / ps))
    n_W = int(np.floor(W / ps))

    for i in range(n_H):
        for j in range(n_W):
            cropped_image = image[i * ps:(i + 1) * ps, j * ps:(j + 1) * ps]
            new_filename = f"{os.path.basename(filename).split('.')[0]}_{i}_{j}.fits"
            save_fits(cropped_image, new_filename, save_dir+'/')
        
def process_image_cropping(file, dir_, ps=256, type_of_image='SCI'):
    """
    Processes an image file by opening it, cropping it into smaller patches, and saving the patches.

    Parameters:
        file (str): The file path of the image to be processed.
        dir_ (str): The directory where the cropped patches will be saved.
        ps (int, optional): The size of the cropped patches (patch size). Defaults to 256.
        type_of_image (str, optional): The type of the image to extract (e.g., 'SCI'). Defaults to 'SCI'.

    Returns:
        None

    Notes:
        - If the image cannot be opened, the function exits without processing.
        - Cropped patches are saved in the specified directory with filenames in the format:
          `<original_filename>_<row_index>_<col_index>.fits`.

    Raises:
        - Logs a warning if the input image cannot be opened or if the save directory cannot be created.
    """
    img = open_fits(file, type_of_image)
    if img is None:
        return 
    img, filepath, ex_time = img
    save_dir = os.path.join('./', dir_)
    crop_image(img, file, save_dir, ps=ps)
    return 

def calculate_image_stats(data: np.ndarray, sigma: float = 3.0, nsigma: float = 2.0, 
                           npixels: int = 10, footprint_radius: int = 10,  maxiters: int = 10, step: int = 5):
    """
    Calculate sigma-clipped statistics and additional background statistics for an image.

    Parameters:
        data (np.ndarray): Input 2D image data.
        sigma (float, optional): Sigma value for sigma-clipping during background statistics calculation. Defaults to 3.0.
        nsigma (float, optional): Threshold level for object detection. Defaults to 2.0.
        npixels (int, optional): Minimum number of connected pixels required to detect a source. Defaults to 10.
        footprint_radius (int, optional): Radius of the circular footprint used for masking detected sources. Defaults to 10.
        maxiters (int, optional): Maximum number of iterations for sigma-clipping. Defaults to 10.
        step (int, optional): Step size for percentile calculations. Defaults to 5.

    Returns:
        tuple:
            - A tuple containing the following:
                - mean (float): Sigma-clipped mean of the background.
                - median (float): Sigma-clipped median of the background.
                - std (float): Sigma-clipped standard deviation of the background.
                - abs_mean (float): Absolute mean of the entire image.
                - mean_light (float): Sigma-clipped mean of the light source regions.
                - median_light (float): Sigma-clipped median of the light source regions.
                - std_light (float): Sigma-clipped standard deviation of the light source regions.
            - mask (np.ndarray): A boolean mask indicating non-source regions (background).
            - percentage_stats (dict): Percentile-based statistics for the image.

    Notes:
        - Light sources are detected using a threshold derived from `nsigma` and masked with a circular footprint.
        - Percentile-based statistics are calculated using the `classify_data` function.

    Exceptions:
        - Returns default values (NaN for statistics, zeros for the mask) if an error occurs during processing.

    Example:
         stats, mask, percentile_stats = calculate_image_stats(image_data)
    """
    if data is None:
        return
    
    try:
        abs_mean = np.mean(data)
        percentage_stats = classify_data(data, step)
        sigma_clip = SigmaClip(sigma=sigma, maxiters=maxiters)
        threshold = detect_threshold(data, nsigma=nsigma, sigma_clip=sigma_clip)
        segment_img = detect_sources(data, threshold, npixels=npixels)
        footprint = circular_footprint(radius=footprint_radius)
        mask = segment_img.make_source_mask(footprint=footprint)
        mean_light, median_light, std_light = sigma_clipped_stats(data, sigma=sigma, mask=mask)
        mask = ~mask
        mean, median, std = sigma_clipped_stats(data, sigma=sigma, mask=mask)
        return (mean, median, std, abs_mean, mean_light, median_light, std_light), np.array(mask).reshape(data.shape), percentage_stats
    except Exception as err:
        ##logging.warning(f'an error occurred while masking the light source: {err}')
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan), np.zeros(data.shape), percentage_stats
    
def classify_data(data, step):
    """
    Classifies data into percentiles and computes statistics for each percentile.

    Args:
        data (array-like): Input numerical data to be classified.
        step (int): Step size for percentile computation (e.g., step=10 for deciles).

    Returns:
        dict: A dictionary containing mean, median, standard deviation, and max 
              statistics for each percentile.
    """
    data_ = data.copy()
    data_ = np.abs(np.asarray(data_)).flatten()
    data_ = np.sort(data_)
    
    n = int(101 / step)
    percentiles = np.linspace(0, 100, n).astype(int)
    
    stats = {}
    for p in percentiles:
        cutoff = len(data) * p // 100
        subset = data[cutoff:]
        stats[f'{p}_mean'] = np.mean(subset)
        stats[f'{p}_median'] = np.median(subset)
        stats[f'{p}_std'] = np.std(subset)
        stats[f'{p}_max'] = np.std(subset)
    return stats
    
    ####################FILTERING AND DOWNLOADING DATA################################

def filter_out_metadata(filepath, col, exp_column, allowed_survey, size=1000, low=100, high=10000, seed=42):
    """
    Filters and samples metadata from a CSV file based on specific conditions.

    Parameters:
        filepath (str): The path to the CSV file containing the metadata.
        col (str): The column name to filter based on allowed survey values.
        exp_column (str): The column name for applying range filtering and sampling.
        allowed_survey (list): A list of allowed values for the `col` column.
        size (int, optional): The desired size of the final filtered dataset. Defaults to 1000.
        low (int, optional): The minimum acceptable value for the `exp_column`. Defaults to 100.
        high (int, optional): The maximum acceptable value for the `exp_column`. Defaults to 10000.
        seed (int, optional): The random seed for shuffling and sampling. Defaults to 42.

    Returns:
        pd.DataFrame: A filtered and sampled DataFrame, sorted by `exp_column`.
                      Returns `None` if errors occur during processing.

    Raises:
        FileNotFoundError: If the specified `filepath` does not exist.
        ValueError: If the required columns (`col` or `exp_column`) are not in the CSV file.

    Notes:
        - The function filters the data to include only rows where `col` has a value in `allowed_survey` 
          and `exp_column` values lie within the specified range (`low` to `high`).
        - It ensures the resulting dataset contains unique values in `exp_column` and attempts 
          to reach the desired size (`size`) by adding samples based on a normal distribution 
          if the initial filtering does not yield enough rows.
    """
    if not os.path.exists(filepath):
        logging.warning(f'no file found at: {filepath}')
        return
    try:
        df = pd.read_csv(filepath)
    except Exception as err:
        logging.warning(f'{filepath} is not a valid csv file: {err}')
        return
    
    if col not in df:
        logging.warning(f'{col} is not in the csv file')
        return  
    if exp_column not in df:
        logging.warning(f'{exp_column} is not in the csv file')
        return  
    
    
    df = df[df[col].isin(allowed_survey)]
    df = df[(df[exp_column] <= high) & (df[exp_column] >= low)]

    df.sample(frac=1, random_state=seed).reset_index(drop=True, inplace=True)
    df['temp_index'] = np.arange(0, len(df))
    part1 = df.drop_duplicates(subset=[exp_column], keep='first')
    part1.reset_index(drop=True, inplace=True)

    if len(part1) < size:
        subdf = df[~df['temp_index'].isin(part1['temp_index'])]
        subdf.sort_values(by=[exp_column], inplace=True)
        subdf.reset_index(drop=True, inplace=True)

        np.random.seed(seed)
        additional_size = size - len(part1)

        indices = np.random.normal(loc=len(subdf) // 2, scale=len(subdf) // 4, size=additional_size)
        unique_indices = np.unique(indices)
        unique_indices = np.clip(unique_indices, 0, len(subdf) - 1).astype(int)

        np.random.shuffle(unique_indices)
        part2 = subdf.iloc[unique_indices]
        part1 = pd.concat([part1, part2], ignore_index=True)

    part1.drop(['temp_index'], inplace=True, axis=1)
    part1.sort_values(by=[exp_column], inplace=True)
    return part1

def test_train_validation_split(dataset_dir, dataset_metadata, url_column, split=(60, 20, 20), seed=42):
    """
    Splits a dataset into training, testing, and validation sets.

    Parameters:
        dataset_dir (str): Directory where the dataset is stored or to be created.
        dataset_metadata (pd.DataFrame): Metadata containing the dataset information.
        url_column (str): The column in the metadata that contains the data URLs or identifiers.
        split (tuple, optional): A tuple of percentages for the train, test, and validation split. 
                                 Defaults to (60, 20, 20).
        seed (int, optional): Seed for reproducibility of the random shuffling and splitting. 
                              Defaults to 42.

    Returns:
        tuple: A tuple containing three arrays:
               - Training data (array)
               - Testing data (array)
               - Validation data (array)

    Raises:
        ValueError: If `url_column` is not in `dataset_metadata`.
    
    Notes:
        - The function assumes the split percentages add up to 100%. No explicit check is performed.
        - Data is randomly shuffled and split based on the provided percentages.
        - The seed ensures reproducibility of the splits.
    """
    
    #if not create_dir(dataset_dir):
    #    return
    
    np.random.seed(seed)
    if url_column not in dataset_metadata:
        logging.warning(f'{url_column} is not in the metadata provided')
        return

    data = dataset_metadata[url_column].to_numpy()
    np.random.shuffle(data)

    total_len = len(data)
    test_size = total_len * split[1] // 100
    val_size = total_len * split[2] // 100

    test_indices = np.random.choice(np.arange(total_len), size=test_size, replace=False)
    rest_indices = [i for i in range(total_len) if i not in test_indices]
    val_indices = np.random.choice(rest_indices, size=val_size, replace=False)
    train_indices = [i for i in range(total_len) if i not in test_indices and i not in val_indices]

    return data[train_indices], data[test_indices], data[val_indices]

def download_dataset(metadata, id_column, url_column, save_dir, max_requests=5, reset_after=10):
    """
    Download images from URLs provided in a metadata DataFrame.

    Parameters:
        metadata (pd.DataFrame): The metadata containing image IDs and URLs.
        id_column (str): The column name in the metadata that contains unique image IDs.
        url_column (str): The column name in the metadata that contains URLs for downloading images.
        save_dir (str): Directory where the downloaded images will be saved.
        max_requests (int, optional): Maximum number of concurrent requests. Defaults to 5.
        reset_after (int, optional): Reset the connection pool after this many requests. Defaults to 10.

    Returns:
        Any: Returns the result of the asynchronous image download operation.
    
    Notes:
        - The function validates the presence of `id_column` and `url_column` in the metadata.
        - Downloads images asynchronously using the `download_images` coroutine.
        - Uses `asyncio.run()` to execute the asynchronous download logic.
    """
    if id_column not in metadata:
        logging.warning(f'{id_column} is not in the metadata provided')
        return  
    if url_column not in metadata:
        logging.warning(f'{url_column} is not in the metadata provided')
        return  
    return asyncio.run(download_images(metadata[id_column], metadata[url_column], save_dir, max_requests, reset_after))

def process_image_stats(file, type_of_image='SCI', sigma=3, nsigma=2, npixels=10, footprint_radius=10, maxiters=10, step=5, save=False):
    """
    Process an image to calculate statistical metrics, including sigma-clipped statistics,
    background, and light source characteristics.

    Parameters:
        file (str): Path to the FITS file to process.
        type_of_image (str, optional): Type of image extension to read (e.g., 'SCI'). Defaults to 'SCI'.
        sigma (float, optional): Sigma value for sigma-clipping during background statistics calculation. Defaults to 3.
        nsigma (float, optional): Threshold level for object detection. Defaults to 2.
        npixels (int, optional): Minimum number of connected pixels required to detect a source. Defaults to 10.
        footprint_radius (int, optional): Radius of the circular footprint used for masking detected sources. Defaults to 10.
        maxiters (int, optional): Maximum number of iterations for sigma-clipping. Defaults to 10.
        step (int, optional): Step size for percentile-based statistics. Defaults to 5.
        save (bool, optional): Whether to save the masked image. Defaults to False.

    Returns:
        dict: A dictionary containing:
            - Filename and location of the file.
            - Sigma-clipped mean, median, and standard deviation of the background.
            - Absolute mean of the entire image.
            - Statistics of light source regions (mean, median, std).
            - Percentile-based statistics.

    Notes:
        - Saves the masked image in a 'masked_images' subdirectory if `save` is True.
        - Ignores NaN, positive infinity, and negative infinity values by converting them to 0.
    """
        
    img = open_fits(file, type_of_image)
    if img is None:
        return 
    img, filepath, ex_time = img
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    (mean, median, std, abs_mean, mean_light, median_light, std_light), mask, percentage_stats = calculate_image_stats(img, sigma, nsigma, npixels, footprint_radius,  maxiters, step)
    if save and not np.all(np.isnan(mask)):
        dir_ = os.path.dirname(file)
        masked_path = os.path.join(dir_, 'masked_images')
        create_dir(masked_path)
        save_fits(img[mask], f'masked_{os.path.basename(file)}', masked_path + '/')
    
    to_be_returned = {'filename':os.path.basename(file), 'location':file, 'mean':mean, 'median':median, 'std':std, 'abs_mean':abs_mean,
                        'light_mean':mean_light, 'light_median':median_light, 'light_std':std_light}
    to_be_returned.update(percentage_stats)
    return to_be_returned

def masked_process_image_stats(file, type_of_image='SCI', sigma=3, nsigma=2, npixels=10, footprint_radius=10, maxiters=10, step=5, save=False):
    """
    Process a pre-masked image to calculate statistical metrics, including sigma-clipped statistics,
    background, and light source characteristics.

    Parameters:
        file (str): Path to the masked FITS file to process.
        type_of_image (str, optional): Type of image extension to read (e.g., 'SCI'). Defaults to 'SCI'.
        sigma (float, optional): Sigma value for sigma-clipping during background statistics calculation. Defaults to 3.
        nsigma (float, optional): Threshold level for object detection. Defaults to 2.
        npixels (int, optional): Minimum number of connected pixels required to detect a source. Defaults to 10.
        footprint_radius (int, optional): Radius of the circular footprint used for masking detected sources. Defaults to 10.
        maxiters (int, optional): Maximum number of iterations for sigma-clipping. Defaults to 10.
        step (int, optional): Step size for percentile-based statistics. Defaults to 5.
        save (bool, optional): Whether to save the processed masked image. Defaults to False.

    Returns:
        dict: A dictionary containing:
            - Filename and location of the file.
            - Sigma-clipped mean, median, and standard deviation of the background.
            - Absolute mean of the entire image.
            - Statistics of light source regions (mean, median, std).
            - Percentile-based statistics.

    Notes:
        - Saves the processed masked image in a 'masked_images' subdirectory if `save` is True.
        - Ignores NaN, positive infinity, and negative infinity values by converting them to 0.
    """
    dir_ = os.path.dirname(file)
    masked_path = os.path.join(dir_, 'masked_images')
    if not os.path.exists(masked_path):
        create_dir(masked_path)

    img = masked_open_fits(file)
    if img is None:
        return 
    img, filepath = img
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    (mean, median, std, abs_mean, mean_light, median_light, std_light), mask, percentage_stats = calculate_image_stats(img, sigma, nsigma, npixels, footprint_radius,  maxiters, step)
    if save and not np.all(np.isnan(mask)):
        save_fits(img[mask], f'masked_{os.path.basename(file)}', masked_path + '/')
    
    to_be_returned = {'filename':os.path.basename(file), 'location':file, 'mean':mean, 'median':median, 'std':std, 'abs_mean':abs_mean,
                        'light_mean':mean_light, 'light_median':median_light, 'light_std':std_light}
    to_be_returned.update(percentage_stats)
    return to_be_returned

def control_flow(dataset_dir, filepath, survey_column, exp_column, id_column, url_column, allowed_survey, download=False, masking=True,
                 masking_cropped=False, save=True, size=1000, low=100, high=1000, seed=42, split=(60, 20, 20), max_requests=5, reset_after=10,
                type_of_image='SCI', sigma=3, nsigma=2, npixels=10, footprint_radius=10, maxiters=10, ps=256, max_workers=16, step=5):
    """
    Orchestrates the data processing and image downloading, including metadata filtering, image cropping, 
    image statistics calculation, and masking. It handles the full pipeline of downloading, processing, 
    and saving image data along with statistics, and optionally applies masking and handles cropped images.

    Parameters:
        dataset_dir (str): Directory where the dataset will be stored.
        filepath (str): Path to the metadata file.
        survey_column (str): Column in the metadata that specifies the survey type.
        exp_column (str): Column in the metadata that specifies the experiment type.
        id_column (str): Column in the metadata that contains unique image IDs.
        url_column (str): Column in the metadata that contains URLs for image downloading.
        allowed_survey (list): List of allowed survey types to filter by.
        download (bool, optional): Flag to enable downloading of images. Defaults to False.
        masking (bool, optional): Flag to enable masking of images. Defaults to True.
        masking_cropped (bool, optional): Flag to enable masking of cropped images. Defaults to False.
        save (bool, optional): Flag to enable saving of processed images and metadata. Defaults to True.
        size (int, optional): Size of the dataset to filter. Defaults to 1000.
        low (int, optional): Lower bound for filtering metadata. Defaults to 100.
        high (int, optional): Upper bound for filtering metadata. Defaults to 1000.
        seed (int, optional): Seed for random number generation. Defaults to 42.
        split (tuple, optional): Tuple to define the dataset split (train, test, validation). Defaults to (60, 20, 20).
        max_requests (int, optional): Maximum number of concurrent requests for image downloading. Defaults to 5.
        reset_after (int, optional): Reset the connection pool after this many requests. Defaults to 10.
        type_of_image (str, optional): Type of image to process ('SCI' by default). Defaults to 'SCI'.
        sigma (float, optional): Sigma value for sigma clipping. Defaults to 3.
        nsigma (float, optional): Threshold level for object detection. Defaults to 2.
        npixels (int, optional): Minimum number of connected pixels for source detection. Defaults to 10.
        footprint_radius (int, optional): Radius of the circular footprint for masking. Defaults to 10.
        maxiters (int, optional): Maximum iterations for sigma clipping. Defaults to 10.
        ps (int, optional): Pixel size for cropping images. Defaults to 256.
        max_workers (int, optional): Maximum number of workers for parallel processing. Defaults to 16.
        step (int, optional): Step size for processing image statistics. Defaults to 5.

    Returns:
        None: The function performs a series of processing steps and saves the results.

    Notes:
        - This function involves several steps including filtering metadata, downloading images, 
          processing images (cropping and calculating statistics), and saving the results.
        - The function uses multi-threading (`ThreadPoolExecutor`) for concurrent processing of images.
        - The function can optionally mask images and handle cropped image statistics.
        - Metadata is filtered based on survey and experiment types, and images are downloaded for training, 
          testing, and validation datasets.
        - Final output includes a CSV file with enriched metadata containing statistics and masking results.
    """
    try:
        create_dir(dataset_dir)
        os.chdir(dataset_dir)
    except Exception as err:
        logging.warning(f'could not change directory to: {dataset_dir}' )
        return
    
    output_filtered_metadata_filename = './filtered_metadata.csv'
    if download:
        filtered_metadata = filter_out_metadata(os.path.join(os.path.dirname(os.getcwd()), filepath), survey_column,
                                                 exp_column, allowed_survey, size=size, low=low, high=high, seed=seed)
        filtered_metadata.to_csv(output_filtered_metadata_filename, index=False)
        if filtered_metadata is None:
            return
    #filtered_metadata = pd.concat([filtered_metadata.iloc[0:19]])
    
        result = test_train_validation_split(dataset_dir, filtered_metadata, url_column, split, seed)
        if result is None:
            return
        training_data, test_data, validation_data = result

        for data, dir_ in zip([training_data, test_data, validation_data], ['training', 'test', 'eval']):
            temp_data = filtered_metadata[filtered_metadata[url_column].isin(data)]
            save_dir = os.path.join('./', os.path.join(dir_, 'originals'))
            if not create_dir(save_dir):
                continue
            result = download_dataset(temp_data, id_column, url_column, save_dir, max_requests=max_requests, reset_after=reset_after)
            if result is None:
                continue
        
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_image_cropping, file, dir_, ps, type_of_image)
                        for file in [os.path.join(save_dir, f) for f in os.listdir(save_dir) if f.endswith('.fits')]]
                
                for future in as_completed(futures):
                    if future is not None:
                        future.result()

    noisy_filtered_metadata_output_file = './metadata_filepath_enriched_with_noise.csv'
    if masking:
        noise_attributes = []
        cropped_image_attributes = []
        for folder in ['./training', './test', './eval']:
            save_dir = os.path.join(folder, 'originals')
            if not os.path.exists(folder) or not os.path.exists(save_dir):
                continue
                
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_image_stats, file, type_of_image, sigma, nsigma, npixels, footprint_radius, maxiters, step, save)
                        for file in [os.path.join(save_dir, f) for f in os.listdir(save_dir) if f.endswith('.fits')]]
                
                for future in as_completed(futures):
                    if future is not None:
                        result = future.result()
                        if result is not None:
                            noise_attributes.append(result)

            if masking_cropped:
                print(folder)
                if not os.path.exists(folder):
                    continue

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(masked_process_image_stats, file, type_of_image, sigma, nsigma, npixels, footprint_radius, maxiters, step)
                            for file in [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.fits')]]
                    for future in as_completed(futures):
                        if future is not None:
                            result = future.result()
                            if result is not None:
                                cropped_image_attributes.append(result)

        noise_attributes = pd.DataFrame(noise_attributes)
        filtered_metadata = pd.read_csv(output_filtered_metadata_filename)
        filtered_metadata['filename'] = filtered_metadata[url_column].apply(lambda url:  url.split('%2F')[-1])
        filtered_metadata = filtered_metadata.merge(noise_attributes, on='filename', how='left')
        filtered_metadata.to_csv(noisy_filtered_metadata_output_file, index=False)

    if masking_cropped:
        # Load cropped image attributes
        cropped_image_attributes = pd.DataFrame(cropped_image_attributes)
        cropped_image_attributes['original_filename'] = cropped_image_attributes['filename'].apply(lambda x: x.split('_')[0] + '_drz.fits')

        # Load filtered metadata and extract original_filename
        filtered_metadata = pd.read_csv(output_filtered_metadata_filename)
        filtered_metadata['original_filename'] = filtered_metadata[url_column].apply(lambda url: url.split('%2F')[-1])

        # Load noisy filtered metadata, extract original_filename, and rename columns
        noisy_filtered_metadata = pd.read_csv(noisy_filtered_metadata_output_file)
        noisy_filtered_metadata['original_filename'] = noisy_filtered_metadata[url_column].apply(lambda url: url.split('%2F')[-1])
        stats_columns = [col for col in noisy_filtered_metadata.columns if any(std in col for std in ['mean', 'median', 'std'])]
        cols = ['original_filename']
        cols.extend(stats_columns)
        noisy_filtered_metadata = noisy_filtered_metadata[cols]
        noisy_filtered_metadata.rename(columns={f'{col}': f'org_{col}' for col in cols if col != 'original_filename'}, inplace=True)

        # Merge datasets
        cropped_image_attributes = filtered_metadata.merge(cropped_image_attributes, on='original_filename', how='left')
        cropped_image_attributes = cropped_image_attributes.merge(noisy_filtered_metadata, on='original_filename', how='left')

        # Save the final result
        cropped_image_attributes.to_csv('./cropped_images_stats.csv', index=False)

def main():

    try:
        directory = os.path.dirname(__file__)
        os.chdir(directory)
    except Exception as err:
        logging.warning(f'an error occurred while changing the working directory to {directory}, {err}')

    dataset_directory = 'nsigma3_fprint_5_npixels_5'
    filepath = 'hst_wfc3_f160W_metadata.csv'
    survey_column = 'sci_aper_1234'
    exp_column = 'sci_actual_duration'
    id_column = 'sci_data_set_name'
    url_column = 'drz_URL'
    allowed_survey = ["IR", "IR-FIX", "IR-UVIS-CENTER", "IR-UVIS-FIX", "GRISM1024", "IR-UVIS", "G141-REF", "G102-REF"]
    download = True
    masking = True
    masking_cropped = True
    save = True
    size = 100
    low = 250
    high = 15000
    seed = 42
    split = (60, 20, 20)
    max_requests = 5
    reset_after = 10
    type_of_image = 'DRZ'
    sigma = 3
    nsigma = 3 
    npixels = 5
    footprint_radius = 5
    maxiters = 25
    ps = 256
    max_workers = 16
    step = 1

    control_flow(dataset_directory, filepath, survey_column, exp_column, id_column, url_column, allowed_survey,
                download, masking, masking_cropped, save,
                size, low, high, seed, split, max_requests, reset_after,
                type_of_image, sigma, nsigma, npixels, footprint_radius, maxiters, ps, max_workers, step)
    
main()
#filtered_metadata = filter_out_metadata(metadata_filepath, col, exp_column, allowed_survey, size=1000, low=100, high=10000, seed=42)
#filtered_metadata.to_csv('./filtered_metadata.csv')
#plot_histogram(filtered_metadata, exp_column, './filtered_exp_column_histogram.png')