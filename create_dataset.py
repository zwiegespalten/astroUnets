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

def create_dir(save_dir):
    try:
        os.makedirs(save_dir, exist_ok=True)
        return True
    except Exception as err:
        logging.warning(f'an error occurred while creating the directory: {save_dir}, {err}')
        return False
    
def save_fits(image, name, path):
    try:
        hdu = fits.PrimaryHDU(image)
        hdul = fits.HDUList([hdu])
        hdul.writeto(path + name, overwrite=True)
        return True
    except Exception as err:
        logging.warning(f'an error occurred while saving the image: {err}')
        return
                        
def plot_histogram(df, exp_column, filename):
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

def open_fits(filepath, type_of_image='SCI'):
    try:
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
            if out is None or ex_time is None:
                return None
            
            return out, filepath, ex_time
    except Exception as err:
        logging.warning(f'Error occurred while reading the .fits file: {err}')
        return None

def masked_open_fits(filepath):
    try:
        with fits.open(filepath) as f:
            out = None
            
            for hdu in f:
                if isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU)) and isinstance(hdu.data, np.ndarray):
                    out = hdu.data
                    break
            
            if out is None:
                return None
            return out, filepath
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None
    
def filter_out_metadata(filepath, col, exp_column, allowed_survey, size=1000, low=100, high=10000, seed=42):
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
    if not create_dir(dataset_dir):
        return
    
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
    if id_column not in metadata:
        logging.warning(f'{id_column} is not in the metadata provided')
        return  
    if url_column not in metadata:
        logging.warning(f'{url_column} is not in the metadata provided')
        return  
    return asyncio.run(download_images(metadata[id_column], metadata[url_column], save_dir, max_requests, reset_after))

def crop_image(image, filename, save_dir, ps=256):
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
    img = open_fits(file, type_of_image)
    if img is None:
        return 
    img, filepath, ex_time = img
    save_dir = os.path.join('./', dir_)
    crop_image(img, file, save_dir, ps=ps)
    return 

def calculate_image_stats(data: np.ndarray, sigma: float = 3.0, nsigma: float = 2.0, 
                           npixels: int = 10, footprint_radius: int = 10,  maxiters: int = 10):
    """
    Calculate sigma-clipped statistics (mean, median, std) of the background in an image.
    Parameters:
        data (np.ndarray): Input 2D image data.
        sigma (float): Sigma for sigma-clipping (default is 3.0).
        nsigma (float): Threshold level for object detection (default is 2.0).
        npixels (int): Minimum number of connected pixels to detect a source (default is 10).
        footprint_radius (int): Radius of the circular footprint used for masking sources (default is 10).
    Returns:
        tuple: A tuple containing (mean, median, std) of the background.
    """
    if data is None:
        return
    
    try:
        abs_mean = np.mean(data)
        sigma_clip = SigmaClip(sigma=sigma, maxiters=maxiters)
        threshold = detect_threshold(data, nsigma=nsigma, sigma_clip=sigma_clip)
        segment_img = detect_sources(data, threshold, npixels=npixels)
        footprint = circular_footprint(radius=footprint_radius)
        mask = segment_img.make_source_mask(footprint=footprint)
        mask = ~mask
        mean, median, std = sigma_clipped_stats(data, sigma=sigma, mask=mask)
        return (mean, median, std, abs_mean), np.array(mask).reshape(data.shape)
    except Exception as err:
        ##logging.warning(f'an error occurred while masking the light source: {err}')
        return (np.nan, np.nan, np.nan, np.nan), np.zeros(data.shape)
    
def process_image_stats(file, type_of_image='SCI', sigma=3, nsigma=2, npixels=10, footprint_radius=10, maxiters=10, save=False):
    img = open_fits(file, type_of_image)
    if img is None:
        return 
    img, filepath, ex_time = img
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    (mean, median, std, abs_mean), mask = calculate_image_stats(img, sigma, nsigma, npixels, footprint_radius,  maxiters)
    if save and not np.all(np.isnan(mask)):
        dir_ = os.path.dirname(file)
        masked_path = os.path.join(dir_, 'masked_images')
        create_dir(masked_path)
        save_fits(img[mask], f'masked_{os.path.basename(file)}', masked_path + '/')
    return {'filename':os.path.basename(file), 'location':file, 'mean':mean, 'median':median, 'std':std, 'abs_mean':abs_mean}

def masked_process_image_stats(file, type_of_image='SCI', sigma=3, nsigma=2, npixels=10, footprint_radius=10, maxiters=10, save=False):
    dir_ = os.path.dirname(file)
    masked_path = os.path.join(dir_, 'masked_images')
    if not os.path.exists(masked_path):
        create_dir(masked_path)

    img = masked_open_fits(file)
    if img is None:
        return 
    img, filepath = img
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    (mean, median, std, abs_mean), mask = calculate_image_stats(img, sigma, nsigma, npixels, footprint_radius,  maxiters)
    if save and not np.all(np.isnan(mask)):
        save_fits(img[mask], f'masked_{os.path.basename(file)}', masked_path + '/')
    return {'filename':os.path.basename(file), 'location':file, 'mean':mean, 'median':median, 'std':std, 'abs_mean':abs_mean}

def control_flow(dataset_dir, filepath, survey_column, exp_column, id_column, url_column, allowed_survey, download=False, masking=True,
                 masking_cropped=False, save=True, size=1000, low=100, high=1000, seed=42, split=(60, 20, 20), max_requests=5, reset_after=10,
                type_of_image='SCI', sigma=3, nsigma=2, npixels=10, footprint_radius=10, maxiters=10, ps=256, max_workers=16):
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
                futures = [executor.submit(process_image_stats, file, type_of_image, sigma, nsigma, npixels, footprint_radius, maxiters, save)
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
                    futures = [executor.submit(masked_process_image_stats, file, type_of_image, sigma, nsigma, npixels, footprint_radius, maxiters)
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
        noisy_filtered_metadata = noisy_filtered_metadata[['original_filename', 'mean', 'median', 'std']]
        noisy_filtered_metadata.rename(columns={'mean': 'org_mean', 'median': 'org_median', 'std': 'org_std'}, inplace=True)

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

    dataset_directory = 'our_data'
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
    type_of_image = 'SCI'
    sigma = 3
    nsigma = 2 
    npixels = 10
    footprint_radius = 10
    maxiters = 25
    ps = 256
    max_workers = 16

    control_flow(dataset_directory, filepath, survey_column, exp_column, id_column, url_column, allowed_survey,
                download, masking, masking_cropped, save,
                size, low, high, seed, split, max_requests, reset_after,
                type_of_image, sigma, nsigma, npixels, footprint_radius, maxiters, ps, max_workers)
    
main()
#filtered_metadata = filter_out_metadata(metadata_filepath, col, exp_column, allowed_survey, size=1000, low=100, high=10000, seed=42)
#filtered_metadata.to_csv('./filtered_metadata.csv')
#plot_histogram(filtered_metadata, exp_column, './filtered_exp_column_histogram.png')