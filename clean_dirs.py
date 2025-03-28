import os
import shutil
from concurrent.futures import ThreadPoolExecutor

def clean_directory(directory):
    """Deletes all files and subdirectories in the given directory."""
    if not os.path.exists(directory):
        print(f"Skipping: {directory} does not exist.")
        return

    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        try:
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)  # Remove directory
            else:
                os.remove(item_path)  # Remove file
        except Exception as e:
            print(f"Error cleaning {item_path}: {e}")

    print(f"Cleaned: {directory}")

def clean_directories_parallel(directories, max_workers=16):
    """Cleans multiple directories in parallel using threads."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(clean_directory, directories)

# Base directory
data_dir = '/mnt/sds-hd/sd17f001/eren/astro/astroUnets'

# List of directories to clean
dirs_to_clean = [
    'metrics_gal_data_150', 'metrics_no_deblend', 'metrics_gpu_no_deblend_gal_data',
    'metrics_no_deblend_nsigma2', 'new_dataset', 'zscale_off_new_gal_data_thres_2_5_8',
    'z_scale_old_gal_data_thres_2_5_8', 'z_scale_new_gal_data_thres_2_5_8',
    'normal_data_cropped_150_scale_corrected', 'normal_data_cropped_150',
    'no_scale_new_gal_data_thres_2_5_8', 'no_crop_metrics_for_gal_data',
    'gal_data_cropped_150', 'gal_data_cropped_150_corrected', 'metrics_for_faber',
    'metrics_for_faber_no_scale_thres', 'metrics_for_faber_no_scale_thres_2_5',
    'metrics_for_faber_no_scale_thres_2_5_normal_data', 'metrics_for_faber_scale_on',
    'metrics_for_faber_scale_on_threshold_2_5', 'metrics_for_faber_scale_on_threshold_2_5_scale_off',
    'metrics_for_faber_z_scale_off_thres_2_0', 'metrics_for_faber_z_scale_off_thres_2_5_8',
    'metrics_for_faber_z_scale_off_thres_2_5_normal_data', 'metrics_for_faber_z_scale_on_thres_2_5_normal_data'
]

# Convert relative paths to absolute paths
dirs_to_clean = [os.path.join(data_dir, x) for x in dirs_to_clean]

# Clean directories in parallel
clean_directories_parallel(dirs_to_clean)
