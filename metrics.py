import os, glob, logging, time, sys, json, shutil, h5py
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import numpy as np
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Ellipse 
from matplotlib.lines import Line2D
import sep
from scipy.spatial import cKDTree
from astropy.visualization import MinMaxInterval, ZScaleInterval
from astropy.stats import SigmaClip
from photutils.segmentation import detect_threshold, detect_sources, deblend_sources, SourceCatalog
from photutils.utils import circular_footprint
from new_train import open_fits_2, save_fits, augment_samples_2, augment_samples, min_max_normalization, inverse_min_max_normalization, adaptive_log_transform_and_normalize, inverse_adaptive_log_transform_and_denormalize, zscore_normalization, inverse_zscore_normalization

logging.basicConfig(level=logging.WARNING)

def crop_image(image, ps=256):
    if not isinstance(image, np.ndarray):
        logging.warning(f"the image is not an 'numpy.ndarray' but {type(image)}")
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
            yield cropped_image

def modify_keras_model(input_file):
    """
    Modifies a Keras model file by removing the 'groups' parameter from Conv2DTranspose layers.

    Parameters:
    - input_file (str): Path to the original Keras model file.
    - output_file (str): Path to save the modified model file.
    """
    # Copy the original file to avoid overwriting
    output_file = input_file.replace('.keras', '.h5')
    shutil.copy2(input_file, output_file)
    
    # Open the copied file in read/write mode
    with h5py.File(output_file, 'r+') as file:
        if 'model_config' in file.attrs:
            model_config = json.loads(file.attrs['model_config'])  # Decode first
            
            # Modify the layers
            for layer in model_config['config']['layers']:
                if layer['class_name'] == 'Conv2DTranspose' and 'groups' in layer['config']:
                    del layer['config']['groups']  # Remove the 'groups' parameter
            
            # Save the modified model configuration back into the file
            file.attrs['model_config'] = json.dumps(model_config)

def generate_gaussian_weights(patch_size, sigma=64):
    if patch_size is None or len(patch_size) != 3:
        print(f'{patch_size} is not a tuple of length 3 but {len(patch_size)}')
        return
    h, w = patch_size[:2]  # Extract height and width, ignore depth
    ax_h = np.linspace(-(h // 2), h // 2, h)
    ax_w = np.linspace(-(w // 2), w // 2, w)
    xx, yy = np.meshgrid(ax_w, ax_h)  # Ensure correct shape order
    weights = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return weights[..., np.newaxis]  # Keep channel dimension

def generate_distance_weights(patch_size):
    if patch_size is None or len(patch_size) != 3:
        print(f'{patch_size} is not a tuple of length 3 but {len(patch_size)}')
        return
    h, w = patch_size[:2]
    ax_h = np.linspace(-1, 1, h)
    ax_w = np.linspace(-1, 1, w)
    xx, yy = np.meshgrid(ax_w, ax_h)
    weights = 1.0 - np.sqrt(xx**2 + yy**2)
    return np.clip(weights, 0, None)[..., np.newaxis]

def pad_image(image, patch_size, stride):
    if patch_size is None or len(patch_size) != 3:
        print(f'{patch_size} is not a tuple of length 3 but {len(patch_size)}')
        return
    if image is None:
        print(f"'image' is not an image: {type(image)}")
        return

    image_size = image.shape
    pad_h = int(np.ceil(image_size[0]/patch_size[0])*patch_size[0] - image_size[0])
    pad_w = int(np.ceil(image_size[1]/patch_size[1])*patch_size[1] - image_size[1])
    pad_h_top = pad_h//2
    pad_h_bottom = pad_h - pad_h_top
    pad_w_left = pad_w//2
    pad_w_right = pad_w - pad_w_left

    padded_divisible_image = np.pad(
        image, 
        ((pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right), (0, 0)),
        mode='constant',
        constant_values=0 
    )
    pad_eq_h = patch_size[0] - stride[0]
    pad_eq_w = patch_size[1] - stride[1]

    padded_eq_divisible_image = np.pad(
        padded_divisible_image, 
        ((pad_eq_h, pad_eq_h), (pad_eq_w, pad_eq_w), (0, 0)),
        mode='constant',
        constant_values=0 
    )
    return padded_eq_divisible_image, (pad_h_top, pad_h_bottom, pad_w_left, pad_w_right, pad_eq_h, pad_eq_w)

def strip_pad(image, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right, pad_eq_h, pad_eq_w):
    if image is None:
        print(f"'image' is not an image: {type(image)}")
        return
    if  (pad_eq_h * 2 + pad_h_top + pad_h_bottom) >= image.shape[0]:
        print(f"image has lower dimension than paddings: {image.shape}") 
        return
    if  (pad_eq_w * 2 + pad_w_left + pad_w_right) >= image.shape[1]:
        print(f"image has lower dimension than paddings: {image.shape}") 
        return

    if pad_eq_h > 0 and pad_eq_w > 0:
        image = image[pad_eq_h:-pad_eq_h, pad_eq_w:-pad_eq_w, :]
    if pad_h_top > 0:
        image = image[pad_h_top:, :, :]
    if pad_h_bottom > 0:
        image = image[:-pad_h_bottom, :, :]
    if pad_w_left > 0:
        image = image[:, pad_w_left:, :]
    if pad_w_right > 0:
        image = image[:, :-pad_w_right, :]
    return image

def sliding_window_generator(padded_image, patch_size=(256, 256, 1), stride=(128, 128, 1)):
    H, W, C = padded_image.shape
    for i in range(0, H - patch_size[0] + 1, stride[0]):
        for j in range(0, W - patch_size[1] + 1, stride[1]):
            patch = padded_image[i:i + patch_size[0], j:j + patch_size[1], :]
            yield patch, (i, j)  # Yield instead of appending

def create_prediction_dataset(image, patch_size=(256, 256, 1), stride=(128, 128, 1), batch_size=128):
    patch_generator = sliding_window_generator(image, patch_size, stride)
    dataset = tf.data.Dataset.from_generator(
        lambda: patch_generator,
        output_signature=(
            tf.TensorSpec(shape=(patch_size[0], patch_size[1], patch_size[2]), dtype=tf.float32), 
            tf.TensorSpec(shape=(2,), dtype=tf.int64)
            )
    )
    return dataset.batch(batch_size)

def sliding_window_inference(image, model, patch_size=(256, 256, 1), stride=(128, 128, 1), weighting='average', batch_size=16):
    image_shape = image.shape
    if any(p > s for p, s in zip(patch_size, image_shape)):
        print(f"Patch size {patch_size} must be smaller than image size {image_shape}.")
        return
    if any(st > s for st, s in zip(stride, image_shape)):
        print(f"Stride {stride} must be smaller than image size {image_shape}.")
        return

    try:
        image_fully_padded, (pad_h_top, pad_h_bottom, pad_w_left, pad_w_right, pad_eq_h, pad_eq_w) = pad_image(image, patch_size, stride)
    except Exception as err:
        logging.warning(f"could not unpack 'pad_image'")
        return
    
    output = np.zeros_like(image_fully_padded, dtype=np.float32)
    weight_matrix = np.zeros_like(image_fully_padded, dtype=np.float32)

    if weighting == 'gaussian':
        weights = generate_gaussian_weights(patch_size)
        if weights is None:
            return 
    elif weighting == 'distance':
        weights = generate_distance_weights(patch_size)
        if weights is None:
            return 
    elif weighting == 'average':
        weights = np.ones(patch_size, dtype=np.float32)
        if weights is None:
            return 
    else:
        print("Weighting must be 'average', 'gaussian', or 'distance'.")
        return
    
    try:
        dataset = create_prediction_dataset(image_fully_padded, patch_size, stride, batch_size)
        for patches, positions in dataset:
            batch_predictions = model.predict(patches)  # Get predictions from the model
            for (i, j), prediction in zip(positions, batch_predictions):
                output[i:i + patch_size[0], j:j + patch_size[1], :] += prediction * weights
                weight_matrix[i:i + patch_size[0], j:j + patch_size[1], :] += weights

        output = strip_pad(output, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right, pad_eq_h, pad_eq_w)
        weight_matrix = strip_pad(weight_matrix, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right, pad_eq_h, pad_eq_w)
    except Exception as err:
        logging.warning(f"an error occurred while stripping padding: {err}")
        return
    return output / np.maximum(weight_matrix, 1e-8)

def scale_image(image, vmin=None, vmax=None):
    try:
        if not isinstance(image, np.ndarray):
            print(f"Expected a NumPy array, got {type(image)}.")
            return
        if image.ndim < 2:
            print(f"Expected at least a 2D image, got shape {image.shape}.")
            return
        if not np.issubdtype(image.dtype, np.number):
            print(f"Image array must contain numerical values, got dtype {image.dtype}.")
            return

        if vmin is None or vmax is None:
            interval = ZScaleInterval()
            vmin, vmax = interval.get_limits(image)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        normalized_data = (norm(image) * 255).astype(np.uint8)
        img = Image.fromarray(normalized_data)
        img_resized = img.resize((img.width, img.height), Image.LANCZOS)
    except Exception as err:
        logging.warning(f"No scaling of images was possible: {err}, returning the original image.")
        return image, vmin, vmax
    return img_resized, vmin, vmax

def plot_source_comparison(original_image, noisy_image, reconstructed_image, 
                           x_org, y_org, x_rec, y_rec, 
                           matched_indices_org, unmatched_indices_org, 
                           matched_indices_rec, unmatched_indices_rec,
                           filename):
    try:
        for image in [original_image, noisy_image, reconstructed_image]:
            if not isinstance(image, np.ndarray):
                print(f"Expected a NumPy array, got {type(image)}.")
                return
            if image.ndim < 2:
                print(f"Expected at least a 2D image, got shape {image.shape}.")
                return
            if not np.issubdtype(image.dtype, np.number):
                print(f"Image array must contain numerical values, got dtype {image.dtype}.")
                return

        try:
            original_image_scaled, vmin, vmax = scale_image(original_image)
            noisy_image_scaled, vmin, vmax = scale_image(noisy_image, vmin=None, vmax=None)
            reconstructed_image_scaled, vmin, vmax = scale_image(reconstructed_image, vmin=None, vmax=None)
        except Exception as err:
            logging.warning(f'an error occurred while scaling images: {err}')
            return

        plt.figure(figsize=(18, 6))

        # Original Image
        plt.subplot(1, 3, 1)
        plt.imshow(original_image_scaled, cmap='gray')
        plt.scatter(x_org[unmatched_indices_org], y_org[unmatched_indices_org], 
                    edgecolors='blue', facecolors='none', label='Unmatched Sources (Original)')
        plt.scatter(x_org[matched_indices_org], y_org[matched_indices_org], 
                    edgecolors='red', facecolors='none', label='Matched Sources')
        plt.title('Detected Sources in Original Image')
        plt.legend()

        # Noisy Image (Just Plot)
        plt.subplot(1, 3, 2)
        plt.imshow(noisy_image_scaled, cmap='gray')
        plt.title('Noisy Image')

        # Reconstructed Image
        plt.subplot(1, 3, 3)
        plt.imshow(reconstructed_image_scaled, cmap='gray')
        plt.scatter(x_rec[unmatched_indices_rec], y_rec[unmatched_indices_rec], 
                    edgecolors='green', facecolors='none', label='Unmatched Sources (Reconstructed)')
        plt.scatter(x_rec[matched_indices_rec], y_rec[matched_indices_rec], 
                    edgecolors='red', facecolors='none', label='Matched Sources')
        plt.title('Detected Sources in Reconstructed Image')
        plt.legend()

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as err:
        logging.warning(f"an error occured while creating combined images: {err}")

def plot_source_comparison_sep(original_image, noisy_image, reconstructed_image, 
                           x_org, y_org, x_rec, y_rec, 
                           matched_indices_org, unmatched_indices_org, 
                           matched_indices_rec, unmatched_indices_rec,
                           org_a, org_b, rec_a, rec_b, org_theta, rec_theta,
                           filename):
    try:
        for image in [original_image, noisy_image, reconstructed_image]:
            if not isinstance(image, np.ndarray):
                print(f"Expected a NumPy array, got {type(image)}.")
                return
            if image.ndim < 2:
                print(f"Expected at least a 2D image, got shape {image.shape}.")
                return
            if not np.issubdtype(image.dtype, np.number):
                print(f"Image array must contain numerical values, got dtype {image.dtype}.")
                return

        try:
            original_image_scaled, vmin, vmax = scale_image(original_image)
            noisy_image_scaled, _, _ = scale_image(noisy_image, vmin=None, vmax=None)
            reconstructed_image_scaled, _, _ = scale_image(reconstructed_image, vmin=None, vmax=None)
        except Exception as err:
            logging.warning(f"An error occurred while scaling images: {err}")
            return

        plt.figure(figsize=(18, 6))

        # Original Image
        ax1 = plt.subplot(1, 3, 1)
        ax1.imshow(original_image_scaled, cmap='gray')

        for x, y, a, b, theta in zip(x_org[unmatched_indices_org], y_org[unmatched_indices_org], 
                                      org_a[unmatched_indices_org], org_b[unmatched_indices_org], 
                                      org_theta[unmatched_indices_org]):
            e = Ellipse(xy=(x, y), width=6*a, height=6*b, angle=np.degrees(theta), 
                        edgecolor='blue', facecolor='none', linewidth=1.5)
            ax1.add_patch(e)

        for x, y, a, b, theta in zip(x_org[matched_indices_org], y_org[matched_indices_org], 
                                      org_a[matched_indices_org], org_b[matched_indices_org], 
                                      org_theta[matched_indices_org]):
            e = Ellipse(xy=(x, y), width=6*a, height=6*b, angle=np.degrees(theta), 
                        edgecolor='red', facecolor='none', linewidth=1.5)
            ax1.add_patch(e)

        ax1.set_title('Detected Sources in Original Image')

        # Noisy Image (Only Image)
        ax2 = plt.subplot(1, 3, 2)
        ax2.imshow(noisy_image_scaled, cmap='gray')
        ax2.set_title('Noisy Image')

        # Reconstructed Image
        ax3 = plt.subplot(1, 3, 3)
        ax3.imshow(reconstructed_image_scaled, cmap='gray')

        for x, y, a, b, theta in zip(x_rec[unmatched_indices_rec], y_rec[unmatched_indices_rec], 
                                      rec_a[unmatched_indices_rec], rec_b[unmatched_indices_rec], 
                                      rec_theta[unmatched_indices_rec]):
            e = Ellipse(xy=(x, y), width=6*a, height=6*b, angle=np.degrees(theta), 
                        edgecolor='green', facecolor='none', linewidth=1.5)
            ax3.add_patch(e)

        for x, y, a, b, theta in zip(x_rec[matched_indices_rec], y_rec[matched_indices_rec], 
                                      rec_a[matched_indices_rec], rec_b[matched_indices_rec], 
                                      rec_theta[matched_indices_rec]):
            e = Ellipse(xy=(x, y), width=6*a, height=6*b, angle=np.degrees(theta), 
                        edgecolor='red', facecolor='none', linewidth=1.5)
            ax3.add_patch(e)

        ax3.set_title('Detected Sources in Reconstructed Image')

        # Custom Legend
        legend_elements = [
            Line2D([0], [0], color='red', lw=2, label='Matched Sources'),
            Line2D([0], [0], color='blue', lw=2, label='Unmatched Sources (Original)'),
            Line2D([0], [0], color='green', lw=2, label='Unmatched Sources (Reconstructed)')
        ]
        
        plt.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.2),
                   ncol=3, frameon=False)

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    except Exception as err:
        logging.warning(f"An error occurred while creating combined images: {err}")
        
def compute_ssim(x, y, alpha=1, beta=1, gamma=1, k1=0.01, k2=0.03):
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or x.shape != y.shape:
        return

    min_x = np.min(x)
    max_x = np.max(x)
    L = max_x - min_x
    if L == 0:
        return np.nan, np.full_like(x, np.nan)  # SSIM is undefined, so return NaN
    
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2
    c3 = c2 / 2
    # Mean
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y
    # Standard Deviation
    sigma_x = np.std(x)
    sigma_y = np.std(y)
    # Variance
    sigma_x_sq = sigma_x ** 2
    sigma_y_sq = sigma_y ** 2
    # Covariance
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))

    # Luminance component: no division by zero
    l = (2 * mu_x * mu_y + c1) / (mu_x_sq + mu_y_sq + c1)
    # Contrast component: no division by zero
    c = (2 * sigma_x * sigma_y + c2) / (sigma_x_sq + sigma_y_sq + c2)
    # Structure component: carefully avoid division by zero
    s = (sigma_xy + c3) / (sigma_x * sigma_y + c3)
    # SSIM formula
    ssim = (l ** alpha) * (c ** beta) * (s ** gamma)
    return np.nanmean(ssim), ssim  # Ignore NaNs in final mean calculation

def calculate_psnr(original, noisy):
    # Ensure the images are of the same shape
    if not isinstance(original, np.ndarray) or not isinstance(noisy, np.ndarray) or original.shape != noisy.shape:
        print("Input images must have the same dimensions")
        return
    # Calculate MSE
    mse = np.mean((original - noisy) ** 2)
    max_pixel = np.max(original)
    # Calculate PSNR
    if mse == 0:
        return float('inf'), mse  # If no error, PSNR is infinite (perfect image)
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr, mse

def aggregate_df(df, flux_rec, flux_org, flux_error_rec, flux_error_org):
    if not isinstance(df, pd.DataFrame):
        print(f"Expected a Pandas DataFrame, got {type(df)}.")
        return

    tp = df.get('TP', pd.Series(0)).sum()
    fp = df.get('FP', pd.Series(0)).sum()
    fn = df.get('FN', pd.Series(0)).sum()
    psnr_rec = df.get('PSNR_rec', pd.Series(np.nan)).mean()
    psnr_noisy = df.get('PSNR_noisy', pd.Series(np.nan)).mean()
    ssim_rec = df.get('SSIM_rec', pd.Series(np.nan)).mean()
    ssim_noisy = df.get('SSIM_noisy', pd.Series(np.nan)).mean()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f_measure = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    if isinstance(flux_org, np.ndarray) and isinstance(flux_rec, np.ndarray) and flux_org.shape == flux_rec.shape:
        rfe = np.mean((flux_org - flux_rec) / flux_org)
    else:
        rfe = 0
    if isinstance(flux_rec, np.ndarray) and isinstance(flux_error_rec, np.ndarray) and flux_rec.shape == flux_error_rec.shape:
        snr_rec = np.mean(flux_rec / flux_error_rec)
    else:
        snr_rec = 0
    if isinstance(flux_org, np.ndarray) and isinstance(flux_error_org, np.ndarray) and flux_org.shape == flux_error_org.shape:
        snr_org = np.mean(flux_org / flux_error_org)
    else:
        snr_org = 0
    ious = df['IoU'].sum()
    unions = df['union'].sum()

    stats = {
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'Precision': precision,
        'Recall': recall,
        'F-measure': f_measure,
        'RFE': rfe,
        'SNR_org': snr_org,
        'SNR_rec': snr_rec,
        'PSNR_rec': psnr_rec,
        'PSNR_noisy': psnr_noisy,
        'SSIM_rec': ssim_rec,
        'SSIM_noisy': ssim_noisy,
        'IoU' : ious / unions if unions != 0.0 else np.nan
    }
    return stats

def detect_sources_in_image(image, kwargs):
    required_params = ['sigma', 'maxiters', 'nsigma', 'npixels', 'nlevels', 'contrast', 'footprint_radius', 'deblend', 'deblend_timeout']
    missing_params = [param for param in required_params if param not in kwargs]
    if missing_params:
        print(f"Missing required parameters: {', '.join(missing_params)}")
        return 
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        print("Image must be a 2D NumPy array.")
        return 

    try:
        sigma_clip = SigmaClip(sigma=kwargs['sigma'], maxiters=kwargs['maxiters'])
        threshold = detect_threshold(image, nsigma=kwargs['nsigma'], sigma_clip=sigma_clip)
        segment_img = detect_sources(image, threshold, npixels=kwargs['npixels'])
        if segment_img is None:
            return None  # No sources detected

        if kwargs['deblend']:
            start_time = time.time()
            try:
                segm_deblended = deblend_sources(image, segment_img, 
                                                 npixels=kwargs['npixels'], 
                                                 nlevels=kwargs['nlevels'], 
                                                 contrast=kwargs['contrast'])
                if time.time() - start_time > kwargs['deblend_timeout']:
                    segm_deblended = segment_img
            except Exception:
                segm_deblended = segment_img
        else:
            segm_deblended = segment_img
        footprint = circular_footprint(radius=kwargs['footprint_radius'])
        mask = segm_deblended.make_source_mask(footprint=footprint)
        catalog = SourceCatalog(image, segm_deblended)
        if len(catalog) == 0:
            return None

        x_centroids = np.array([source.xcentroid for source in catalog])
        y_centroids = np.array([source.ycentroid for source in catalog])
        fluxes = np.array([source.segment_flux for source in catalog])
        segment_area = np.array([source.segment_area.value for source in catalog])
        background_noise = np.median(image[mask == 0])
        flux_errors = background_noise * np.sqrt(segment_area)
        return x_centroids, y_centroids, fluxes, flux_errors, mask
    except Exception as err:
        print(f"Error during source detection: {err}")
        return
    
def extract_sources(image, flag, kwargs):
    """Extract sources from an astronomical image using SEP and classify them as stars or galaxies."""
    
    # Get parameters from kwargs or set defaults

    thresh = kwargs.get('thresh', 2)  if flag != 'original' else kwargs.get('org_thresh', 1.8)
    radius_factor = kwargs.get('radius_factor', 6.0)  # Factor for Kron radius
    PHOT_FLUXFRAC  = kwargs.get('PHOT_FLUXFRAC ', 0.5)  # Fraction for flux radius
    r_min = kwargs.get('r_min', 1.75)  # Minimum diameter for circular apertures
    elongation_fraction = kwargs.get('elongation_fraction', 1.5)
    PHOT_AUTOPARAMS = kwargs.get('PHOT_AUTOPARAMS', 2.5)

    maskthresh = kwargs.get('maskthresh', 0.0)  # Threshold for pixel masking
    minarea = kwargs.get('minarea', 10)  if flag != 'original' else kwargs.get('org_minarea', 5)
    filter_type = kwargs.get('filter_type', 'matched')  # Filter treatment type
    deblend_nthresh = kwargs.get('deblend_nthresh', 32)  # Number of thresholds for deblending
    deblend_cont = kwargs.get('deblend_cont', 0.005)  # Minimum contrast ratio for deblending
    clean = kwargs.get('clean', True)  # Perform cleaning
    clean_param = kwargs.get('clean_param', 1.0)  # Cleaning parameter

    # Subtract background using SEP
    image = image.astype(image.dtype.newbyteorder('='))  # Converts to the native byte order
    bkg = sep.Background(image)
    data_sub = image - bkg

    objects = sep.extract(data_sub, thresh, err=bkg.globalrms, maskthresh=maskthresh, minarea=minarea, 
                          filter_type=filter_type, deblend_nthresh=deblend_nthresh, deblend_cont=deblend_cont, 
                          clean=clean, clean_param=clean_param)
    
    objects = pd.DataFrame(objects)
    objects['index'] = range(len(objects))

    # Calculate Kron radius
    kronrad, krflag = sep.kron_radius(data_sub, objects['x'], objects['y'], objects['a'], objects['b'], objects['theta'], radius_factor)
    
    # Calculate flux using elliptical aperture
    flux, fluxerr, flag = sep.sum_ellipse(data_sub, objects['x'], objects['y'], objects['a'], objects['b'], objects['theta'], 
                                          PHOT_AUTOPARAMS*kronrad, subpix=1, err=bkg.globalrms)
    flag |= krflag

    # Use circular aperture if Kron radius is small
    use_circle = kronrad * np.sqrt(objects['a'] * objects['b']) < r_min
    cflux, cfluxerr, cflag = sep.sum_circle(data_sub, objects['x'][use_circle], objects['y'][use_circle], r_min, subpix=1, err=bkg.globalrms)
    flux[use_circle] = cflux
    fluxerr[use_circle] = cfluxerr
    flag[use_circle] = cflag

    # Compute flux radius
    r, flag = sep.flux_radius(data_sub, objects['x'], objects['y'], radius_factor*objects['a'], PHOT_FLUXFRAC, normflux=flux, subpix=5)
    
    # Create mask
    mask = np.zeros(data_sub.shape, dtype=bool)
    sep.mask_ellipse(mask, objects['x'], objects['y'], objects['a'], objects['b'], objects['theta'], r=3.)

    # Prepare results
    results = pd.DataFrame({
        'index': objects['index'],
        'kron_radius': kronrad,
        'flux': flux,
        'flux_err': fluxerr,
        'flux_radius': r,
        'sep_flag': flag
    })

    merged_df = pd.merge(objects, results, on="index").drop(columns=['index'])
    merged_df['is_galaxy'] = (merged_df['a'] / merged_df['b'] >= elongation_fraction).astype(int)
    return merged_df, mask

def wrap_extract_sources(image, flag, kwargs):
    df, mask = extract_sources(image, flag, kwargs)
    return df['x'].to_numpy(), df['y'].to_numpy(), df['flux_y'].to_numpy(), df['flux_err'].to_numpy(), mask, df['a'].to_numpy(), df['b'].to_numpy(), df['theta'].to_numpy(), df

def calculate_iou(org_mask, rec_mask):
    """
    Calculate Intersection over Union (IoU) for two binary masks.

    Parameters:
    - org_mask (numpy.ndarray): Binary mask from the original image (1 for source, 0 for background).
    - rec_mask (numpy.ndarray): Binary mask from the reconstructed image (1 for source, 0 for background).

    Returns:
    - IoU (float): The intersection over union (IoU) score between the two masks.
    """
    # Ensure that the masks are binary (values should be either 0 or 1)
    org_mask = np.array(org_mask, dtype=bool)
    rec_mask = np.array(rec_mask, dtype=bool)
    if org_mask.shape != rec_mask.shape:
        return np.nan, np.nan

    intersection = np.sum(org_mask & rec_mask)
    union = np.sum(org_mask | rec_mask)
    iou = intersection / union if union != 0 else 0.0  # Avoid division by zero
    return iou, union

def compare_images(image_org, noisy_image, image_reconstructed, image_id, exp_time,
                    new_exp_time, output_dir, kwargs, if_selected):
    if not isinstance(image_org, np.ndarray) or not isinstance(noisy_image, np.ndarray) or not isinstance(image_reconstructed, np.ndarray):
        return
    if image_org.shape != noisy_image.shape or noisy_image.shape != image_reconstructed.shape or image_org.shape != image_reconstructed.shape:
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    filename = os.path.join(output_dir, f'{image_id}_{round(exp_time)}_{round(new_exp_time)}.png')
    # Detect sources in original and reconstructed images
    try:
        if kwargs['func'] == wrap_extract_sources:
            x_org, y_org, flux_org, flux_error_org, org_mask, org_a, org_b, org_theta, org_df = kwargs['func'](image_org, 'original', kwargs)
            x_rec, y_rec, flux_rec, flux_error_rec, rec_mask, rec_a, rec_b, rec_theta, rec_df = kwargs['func'](image_reconstructed, 'reconstructed', kwargs)
            x_noisy, y_noisy, flux_noisy, flux_error_noisy, noisy_mask, noisy_a, noisy_b, noisy_theta, noisy_df = kwargs['func'](noisy_image, 'noisy', kwargs)
        else:
            x_org, y_org, flux_org, flux_error_org, org_mask = kwargs['func'](image_org, kwargs)
            x_rec, y_rec, flux_rec, flux_error_rec, rec_mask = kwargs['func'](image_reconstructed, kwargs)
            x_noisy, y_noisy, flux_noisy, flux_error_noisy, noisy_mask = kwargs['func'](noisy_image, kwargs)
    except Exception as err:
        logging.warning(f'could not find light sources: {err}')
        return
    
    # Remove NaNs from x_rec and y_rec while keeping valid indices
    valid_rec = ~(np.isnan(x_rec) | np.isnan(y_rec))
    x_rec_clean = x_rec[valid_rec]
    y_rec_clean = y_rec[valid_rec]
    valid_org = ~(np.isnan(x_org) | np.isnan(y_org))
    x_org_clean = x_org[valid_org]
    y_org_clean = y_org[valid_org]
    valid_noisy = ~(np.isnan(x_noisy) | np.isnan(y_noisy))
    x_noisy_clean = x_noisy[valid_noisy]
    y_noisy_clean = y_noisy[valid_noisy]

    if kwargs['func'] == wrap_extract_sources:
        org_df = org_df[valid_org].reset_index(drop=True)
        rec_df = rec_df[valid_rec].reset_index(drop=True)
        noisy_df = noisy_df[valid_noisy].reset_index(drop=True)
        org_df['image_id'] = [image_id]*len(org_df)
        rec_df['image_id'] = [image_id]*len(rec_df)
        noisy_df['image_id'] = [image_id]*len(noisy_df)
        org_df['exp_time'] = [exp_time]*len(org_df)
        noisy_df['exp_time'] = [exp_time]*len(noisy_df)
        noisy_df['new_exp_time'] = [new_exp_time]*len(noisy_df)
        rec_df['exp_time'] = [exp_time]*len(rec_df)
        rec_df['new_exp_time'] = [new_exp_time]*len(rec_df)
    else:
        org_df = rec_df = noisy_df = pd.DataFrame()

    # Build KD-trees with cleaned x_rec and y_rec
    try:
        tree_org = cKDTree(np.column_stack((x_org_clean, y_org_clean)))
        tree_rec = cKDTree(np.column_stack((x_rec_clean, y_rec_clean)))
        matches = tree_rec.query(np.column_stack((x_org_clean, y_org_clean)), distance_upper_bound=kwargs['distance_threshold'])
        matched_indices_image_org = np.where(matches[0] < kwargs['distance_threshold'])[0]
        matched_indices_image_rec = matches[1][matched_indices_image_org]
        unmatched_indices_image_org = np.setdiff1d(np.arange(len(x_org)), matched_indices_image_org)
        unmatched_indices_image_rec = np.setdiff1d(np.arange(len(x_rec)), matched_indices_image_rec)
    except Exception as err:
        print(f"an error occurred while finding mutual sources: {err}")
        return

    try:
        mean_ssmi, ssim = compute_ssim(image_org, image_reconstructed,
                                        kwargs['alpha'], kwargs['beta'],
                                        kwargs['gamma'], kwargs['k1'],
                                        kwargs['k2'])
        mean_noisy_ssmi, noisy_ssim = compute_ssim(image_org, noisy_image, 
                                        kwargs['gamma'], kwargs['k1'],
                                        kwargs['k2'])
        psnr, mse = calculate_psnr(image_org, image_reconstructed)
        noisy_psnr, noisy_mse = calculate_psnr(image_org, noisy_image)
    except Exception as err:
        print(f"an error occurred while calculating the metrics: {err}")

    # Metrics Calculation
    tp = len(matched_indices_image_org)
    fp = len(unmatched_indices_image_rec)
    fn = len(unmatched_indices_image_org)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f_measure = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    rec_error_mask = flux_error_rec[matched_indices_image_rec] != 0.0
    org_error_mask = flux_error_org[matched_indices_image_org] != 0.0 
    flux_rec = flux_rec[matched_indices_image_rec][rec_error_mask]  
    flux_error_rec = flux_error_rec[matched_indices_image_rec][rec_error_mask]
    flux_org = flux_org[matched_indices_image_org][org_error_mask]  
    flux_error_org = flux_error_org[matched_indices_image_org][org_error_mask]

    if kwargs['func'] == wrap_extract_sources:
        org_df = org_df.iloc[matched_indices_image_org].reset_index(drop=True)
        rec_df = rec_df.iloc[matched_indices_image_rec].reset_index(drop=True)

    if isinstance(flux_org, np.ndarray) and isinstance(flux_rec, np.ndarray) and flux_org.shape == flux_rec.shape:
        rfe = np.mean((flux_org - flux_rec) / flux_org)
    else:
        rfe = 0
    if isinstance(flux_rec, np.ndarray) and isinstance(flux_error_rec, np.ndarray) and flux_rec.shape == flux_error_rec.shape:
        snr_rec = np.mean(flux_rec / flux_error_rec)
    else:
        snr_rec = 0
    if isinstance(flux_org, np.ndarray) and isinstance(flux_error_org, np.ndarray) and flux_org.shape == flux_error_org.shape:
        snr_org = np.mean(flux_org / flux_error_org)
    else:
        snr_org = 0
    
    iou, union = calculate_iou(org_mask, rec_mask)
    stats = {
        'image_id': image_id,
        'org_exp_time': exp_time,
        'new_exp_time': new_exp_time,
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'Precision': precision,
        'Recall': recall,
        'F-measure': f_measure,
        'RFE': rfe,
        'SNR_org': snr_org,
        'SNR_rec': snr_rec,
        'PSNR_rec': psnr,
        'PSNR_noisy': noisy_psnr,
        'SSIM_rec': mean_ssmi,
        'SSIM_noisy': mean_noisy_ssmi,
        'IoU' : iou,
        'union' : union
    }
    # Plot source comparison and save the image
    try:
        if if_selected:
            if kwargs['func'] == wrap_extract_sources:
                plot_source_comparison_sep(image_org, noisy_image, image_reconstructed, 
                                x_org, y_org, x_rec, y_rec, 
                                matched_indices_image_org, unmatched_indices_image_org, 
                                matched_indices_image_rec, unmatched_indices_image_rec,
                                org_a, org_b, rec_a, rec_b, org_theta, rec_theta,
                                filename)
            else:
                plot_source_comparison(image_org, noisy_image, image_reconstructed, 
                                x_org, y_org, x_rec, y_rec, 
                                matched_indices_image_org, unmatched_indices_image_org, 
                                matched_indices_image_rec, unmatched_indices_image_rec,
                                filename) 
    except:
        pass
    
    return stats, (flux_rec.flatten().tolist(),
                   flux_org.flatten().tolist()
                   ),(flux_error_rec.flatten().tolist(),
                  flux_error_org.flatten().tolist()
                  ), (org_df, noisy_df, rec_df)

def decide_scale(model_dir):
    if 'log_min_max' in model_dir:
        return (adaptive_log_transform_and_normalize, inverse_adaptive_log_transform_and_denormalize)
    elif 'min_max' in model_dir:
        return (min_max_normalization, inverse_min_max_normalization)
    elif 'z_scale' or 'zscale' in model_dir:
        return (zscore_normalization, inverse_zscore_normalization)
    else:
        return None

def find_best_performing_models(models_dir, condition, filter_model, n):
    models = [f for f in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, f)) and condition(f)]
    print(models)
    my_dict = {}
    for model in models:
        try:
            keras_models = filter_model(glob.glob(os.path.join(models_dir, model, 'checkpoints/*.keras')), n)
            my_dict[model] = keras_models
        except Exception as err:
            print(f'an error occurred while retrieving model names: {err}')
    return my_dict

def get_test_images(N, metadata_filepath, seed=None, dataset_col='', pep_id_col='', location_col='', signal_col='', sigma_col='', 
                    exp_col='', low=200, high=20000, factor=10):
    if not os.path.exists(metadata_filepath):
        print(f"No file found at: {metadata_filepath}")
        return
    try:
        metadata = pd.read_csv(metadata_filepath)
        required_cols = [dataset_col, pep_id_col, location_col, signal_col, sigma_col, exp_col]
        metadata = metadata[required_cols]
    except KeyError as e:
        print(f"Missing expected column: {e}")
        return
    required_columns = [dataset_col, pep_id_col, location_col, signal_col, sigma_col, exp_col]
    missing_cols = [col for col in required_columns if col not in metadata.columns]
    if missing_cols:
        print(f"Missing required columns in DataFrame: {', '.join(missing_cols)}")
        return

    metadata.dropna(inplace=True)
    eval_metadata = metadata[metadata[location_col].str.contains('eval', na=False)]
    eval_metadata = eval_metadata[(eval_metadata[exp_col] >= low * 2) & (eval_metadata[exp_col] <= high)]
    eval_metadata['index'] = range(len(eval_metadata))
    eval_metadata['org_SNR'] = eval_metadata[signal_col] / eval_metadata[sigma_col]
    eval_metadata = eval_metadata[~eval_metadata['org_SNR'].isna()]
    
    add_info = []
    for index, row in eval_metadata.iterrows():
        results = augment_samples_2(row[sigma_col], low, row[exp_col], n_samples_per_magnitude=2)
        if results is None:
            continue
        for result in results:
            if result is not None:
                noisy_sigma, _, _, new_exp_time = result
                combined_sigma = np.sqrt(row[sigma_col] ** 2 + noisy_sigma ** 2)
                add_info.append({
                    dataset_col: row[dataset_col], 
                    'noisy_sigma': noisy_sigma, 
                    'combined_sigma': combined_sigma,
                    "new_exp_time": new_exp_time, 
                    'noisy_SNR': row[signal_col] / combined_sigma
                })
                
    add_info = pd.DataFrame(add_info)
    eval_metadata = pd.merge(eval_metadata, add_info, on=dataset_col)
    eval_metadata['SNR_ratio'] = eval_metadata['noisy_SNR'] / eval_metadata['org_SNR']
    eval_metadata = eval_metadata[eval_metadata['SNR_ratio'] >= 1 / factor]
    N = min(N, len(eval_metadata))
    selected_data = eval_metadata.sample(n=N, random_state=seed).drop(labels=['index'], axis=1)
    return selected_data.sort_values(by=[dataset_col, 'SNR_ratio']).reset_index(drop=True)

def process_single_model(model_file, model_dir, data_dir, scales, test_images_df, output_filedir, kwargs_source, 
                         patch_size=(256, 256, 1), stride=(128, 128, 1), weighting='gaussian', batch_size=16, frac=0.1, model_index=0):
    
    epoch = os.path.basename(model_file).split('.')[0].split('_')[-1]
    required_columns = ['location', 'sci_actual_duration', 'new_exp_time', 'noisy_sigma']
    if not all(col in test_images_df.columns for col in required_columns):
        print(f"DataFrame is missing required columns: {', '.join(required_columns)}")
        return
    try:
        modify_keras_model(model_file)
        model = tf.keras.models.load_model(model_file.replace('.keras', '.h5'), compile=False)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        model.compile(optimizer=optimizer, loss="mae")
    except Exception as err:
        print(f'An exception occurred while loading model: {err} from {model_file}')
        return

    metrics = []
    aggregated_metrics = []
    all_flux_rec = []
    all_flux_org = []
    all_flux_error_rec = [] 
    all_flux_error_org = []
    org_dfs = []
    noisy_dfs = []
    rec_dfs = []

    for _, row in test_images_df.iterrows():
        image_filepath = row['location']
        org_exp_time = row['sci_actual_duration']
        new_exp_time = row['new_exp_time']
        noisy_sigma = row['noisy_sigma']
        if_selected = np.random.rand() <= frac
        
        # Ensure the image filepath is valid (at least 2 characters long to slice)
        if len(image_filepath) > 2:
            image_filepath = f'{data_dir}/{image_filepath[2:]}'
        else:
            print(f"Skipping invalid filepath: {image_filepath}")
            continue
        try:
            image = open_fits_2(image_filepath)
        except Exception as e:
            print(f"Error reading image {image_filepath}: {e}")
            continue

        if image is not None:
            image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
            noisy_image = image + np.random.normal(loc=0, scale=noisy_sigma, size=image.shape)
            #noisy_image -= np.min(noisy_image)
            noisy_image = np.nan_to_num(noisy_image, nan=0.0, posinf=0.0, neginf=0.0)
            reconstructed_image = None 

            if scales is not None:
                scale, descale = scales
                try:
                    args = scale(noisy_image)
                    if args is not None:
                        scaled_noisy_image = args[0]
                        func_args = args[1:]
                        try:
                            reconstructed_image = sliding_window_inference(np.expand_dims(scaled_noisy_image, axis=(-1)), model,
                                                                        patch_size, stride, weighting, batch_size)
                        except Exception as err:
                            logging.warning(f"an error occurred while trying to create mosaics: {err}")
                            continue
                        #reconstructed_image = np.random.randint(0, 10, size=(256, 256, 1))
                        if reconstructed_image is not None:
                            reconstructed_image = descale(reconstructed_image[:, :, 0], *func_args)
                except Exception as e:
                    print(f"Error in scaling or reconstruction: {e}")
                    continue
            else: 
                try:
                    reconstructed_image = sliding_window_inference(np.expand_dims(noisy_image, axis=(-1)), model,
                                                               patch_size, stride, weighting, batch_size)
                except Exception as err:
                    logging.warning(f"an error occurred while trying to create mosaics: {err}")
                    continue

                #reconstructed_image = np.random.randint(0, 10, size=(256, 256, 1))
                if reconstructed_image is not None:
                    reconstructed_image = reconstructed_image[:, :, 0]

            if reconstructed_image is None:
                print(f"Skipping {image_filepath} due to failed reconstruction.")
                continue
            reconstructed_image = np.nan_to_num(reconstructed_image, nan=0.0, posinf=0.0, neginf=0.0)
            #min_rec = np.min(reconstructed_image)
            #if min_rec < 0:
            #    reconstructed_image -= min_rec

            # Create output directories if they don't exist
            combined_image_dir = os.path.join(output_filedir, 'combined_images', str(model_dir), str(epoch))
            png_dir = os.path.join(output_filedir, "pngs")
            org_dir = os.path.join(output_filedir, "original_images")
            noisy_dir = os.path.join(output_filedir, "noisy_images")
            rec_dir = os.path.join(output_filedir, "reconstructed_images")
            output_filedir_model = os.path.join(output_filedir, str(model_dir), str(epoch))
            try:
                for folder in [combined_image_dir, png_dir, output_filedir_model, org_dir, noisy_dir, rec_dir]:
                    if not os.path.exists(folder):
                        os.makedirs(folder, exist_ok=True)
            except Exception as e:
                print(f"Error creating output directories: {e}")
                continue
            # Save FITS and PNG images for original, noisy, and reconstructed
            org_name = os.path.basename(image_filepath).rsplit('.', 1)[0]
            for index, (img, suffix) in enumerate([(image, "_org"),
                                      (reconstructed_image, "_rec"),
                                      (noisy_image, "_noisy")]):
                try:
                    if index == 0:
                        filename = f"{org_name}{suffix}_{round(org_exp_time)}" #original
                        if if_selected:
                            save_fits(img, filename, org_dir + "/")
                    elif index == 1:
                        filename = f"{org_name}{suffix}_{model_dir}_{epoch}_{round(org_exp_time)}_{round(new_exp_time)}" #reconstructed
                        if if_selected:
                            save_fits(img, filename, output_filedir_model + "/")
                            save_fits(img, filename, rec_dir + "/")
                    else:
                        filename = f"{org_name}{suffix}_{round(org_exp_time)}_{round(new_exp_time)}" #noisy
                        if if_selected:
                            save_fits(img, filename, noisy_dir + "/")
                    try:
                        scaled_img, vmin, vmax = scale_image(img)
                        scaled_img.save(os.path.join(png_dir, f"{filename}.png"))
                    except:
                        pass
                except Exception as e:
                    print(f"Error saving images for {org_name}{suffix}: {e}")
                    continue
            
            results = compare_images(image, noisy_image, reconstructed_image, org_name, float(org_exp_time),
                                                 float(new_exp_time), combined_image_dir, kwargs_source, if_selected)
            if results is not None:
                stats, (flux_rec, flux_org), (flux_error_rec, flux_error_org), (org_df, noisy_df, rec_df) = results 
                stats['epoch'] = epoch
                stats['model'] = model_dir
                
                all_flux_rec.extend(flux_rec)
                all_flux_org.extend(flux_org)
                all_flux_error_rec.extend(flux_error_rec)
                all_flux_error_org.extend(flux_error_org)
                metrics.append(stats)

                if not rec_df.empty:
                    if model_index == 0:
                        org_dfs.append(org_df)
                        noisy_dfs.append(noisy_df)
                    rec_dfs.append(rec_df)

    metrics = pd.DataFrame(metrics)
    all_flux_rec = np.array(all_flux_rec)
    all_flux_org = np.array(all_flux_org)
    all_flux_error_rec = np.array(all_flux_error_rec)
    all_flux_error_org = np.array(all_flux_error_org)

    if len(rec_dfs):
        if model_index == 0:
            org_dfs = pd.concat(org_dfs, ignore_index=True)
            noisy_dfs = pd.concat(noisy_dfs, ignore_index=True)
            org_dfs['epoch'] = epoch
            org_dfs['model'] = model_dir
            noisy_dfs['epoch'] = epoch
            noisy_dfs['model'] = model_dir
        else:
            org_dfs = pd.DataFrame(org_dfs)
            noisy_dfs = pd.DataFrame(noisy_dfs)
        rec_dfs = pd.concat(rec_dfs, ignore_index=True)
        rec_dfs['epoch'] = epoch
        rec_dfs['model'] = model_dir
    else:
        org_dfs = pd.DataFrame(org_dfs)
        noisy_dfs = pd.DataFrame(noisy_dfs)
        rec_dfs = pd.DataFrame(rec_dfs)
    try:
        aggregated_stats = aggregate_df(metrics, all_flux_rec, all_flux_org, all_flux_error_rec, all_flux_error_org)
        aggregated_stats['epoch'] = epoch
        aggregated_stats['model'] = model_dir
        aggregated_metrics.append(aggregated_stats)
    except Exception as err:
        logging.warning(f"{err}")
    return metrics, pd.DataFrame(aggregated_metrics), (org_dfs, noisy_dfs, rec_dfs)

def process_single_model_no_mosaic(model_file, model_dir, data_dir, scales, test_images_df, output_filedir, kwargs_source, 
                         patch_size=(256, 256, 1), stride=(128, 128, 1), weighting='gaussian', batch_size=16, frac=0.1, model_index=0):
    
    epoch = os.path.basename(model_file).split('.')[0].split('_')[-1]
    required_columns = ['location', 'sci_actual_duration', 'new_exp_time', 'noisy_sigma']
    if not all(col in test_images_df.columns for col in required_columns):
        print(f"DataFrame is missing required columns: {', '.join(required_columns)}")
        return
    try:
        modify_keras_model(model_file)
        model = tf.keras.models.load_model(model_file.replace('.keras', '.h5'), compile=False)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        model.compile(optimizer=optimizer, loss="mae")
    except Exception as err:
        print(f'An exception occurred while loading model: {err} from {model_file}')
        return

    metrics = []
    aggregated_metrics = []
    all_flux_rec = []
    all_flux_org = []
    all_flux_error_rec = [] 
    all_flux_error_org = []
    org_dfs = []
    noisy_dfs = []
    rec_dfs = []

    for _, row in test_images_df.iterrows():
        image_filepath = row['location']
        org_exp_time = row['sci_actual_duration']
        new_exp_time = row['new_exp_time']
        noisy_sigma = row['noisy_sigma']
        
        # Ensure the image filepath is valid (at least 2 characters long to slice)
        if len(image_filepath) > 2:
            image_filepath = f'{data_dir}/{image_filepath[2:]}'
        else:
            print(f"Skipping invalid filepath: {image_filepath}")
            continue
        try:
            image = open_fits_2(image_filepath)
        except Exception as e:
            print(f"Error reading image {image_filepath}: {e}")
            continue

        if image is None:
            continue

        i = 0
        for cropped_image in crop_image(image):
            if_selected = np.random.rand() <= frac
            cropped_image = np.nan_to_num(cropped_image, nan=0.0, posinf=0.0, neginf=0.0)
            noisy_image = cropped_image + np.random.normal(loc=0, scale=noisy_sigma, size=cropped_image.shape)
            #noisy_image -= np.min(noisy_image)
            noisy_image = np.nan_to_num(noisy_image, nan=0.0,  posinf=0.0, neginf=0.0)
            reconstructed_image = None 
            if scales is not None:
                scale, descale = scales
                try:
                    args = scale(noisy_image)
                    if args is not None:
                        scaled_noisy_image = args[0]
                        func_args = args[1:]
                        try:
                            input_image = np.array([np.expand_dims(scaled_noisy_image, axis=(-1))])
                            reconstructed_image = model.predict(input_image)
                            reconstructed_image = reconstructed_image[0, :, :, 0]
                        except Exception as err:
                            logging.warning(f"an error occurred while trying to create mosaics: {err}")
                            continue
                        if reconstructed_image is not None:
                            reconstructed_image = descale(reconstructed_image, *func_args)
                except Exception as e:
                    print(f"Error in scaling or reconstruction: {e}")
                    continue
            else: 
                try:
                    input_image = np.array([np.expand_dims(noisy_image, axis=(-1))])
                    reconstructed_image = model.predict(input_image)
                    reconstructed_image = reconstructed_image[0, :, :, 0]
                except Exception as err:
                    logging.warning(f"an error occurred while trying to create mosaics: {err}")
                    continue
            if reconstructed_image is None:
                print(f"Skipping {image_filepath} due to failed reconstruction.")
                continue
            
            reconstructed_image = np.nan_to_num(reconstructed_image, nan=0.0,  posinf=0.0, neginf=0.0)
            #min_rec = np.min(reconstructed_image)
            #if min_rec < 0:
            #    reconstructed_image -= min_rec

            # Create output directories if they don't exist
            combined_image_dir = os.path.join(output_filedir, 'combined_images', str(model_dir), str(epoch))
            png_dir = os.path.join(output_filedir, "pngs")
            org_dir = os.path.join(output_filedir, "original_images")
            noisy_dir = os.path.join(output_filedir, "noisy_images")
            rec_dir = os.path.join(output_filedir, "reconstructed_images")
            output_filedir_model = os.path.join(output_filedir, str(model_dir), str(epoch))
            try:
                for folder in [combined_image_dir, png_dir, output_filedir_model, org_dir, noisy_dir, rec_dir]:
                    if not os.path.exists(folder):
                        os.makedirs(folder, exist_ok=True)
            except Exception as e:
                print(f"Error creating output directories: {e}")
                continue
            # Save FITS and PNG images for original, noisy, and reconstructed
            org_name = os.path.basename(image_filepath).rsplit('.', 1)[0]
            for index, (img, suffix) in enumerate([(cropped_image, "_org"),
                                      (reconstructed_image, "_rec"),
                                      (noisy_image, "_noisy")]):
                try:
                    if index == 0:
                        filename = f"{org_name}{suffix}_{round(org_exp_time)}" #original
                        if if_selected:
                            save_fits(img, filename, org_dir + "/")
                    elif index == 1:
                        filename = f"{org_name}{suffix}_{model_dir}_{epoch}_{round(org_exp_time)}_{round(new_exp_time)}" #reconstructed
                        if if_selected:
                            save_fits(img, filename, output_filedir_model + "/")
                            save_fits(img, filename, rec_dir + "/")
                    else:
                        filename = f"{org_name}{suffix}_{round(org_exp_time)}_{round(new_exp_time)}" #noisy
                        if if_selected:
                            save_fits(img, filename, noisy_dir + "/")
                    try:
                        scaled_img, vmin, vmax = scale_image(img)
                        scaled_img.save(os.path.join(png_dir, f"{filename}.png"))
                    except:
                        pass
                except Exception as e:
                    print(f"Error saving images for {org_name}{suffix}: {e}")
                    continue
            
            results = compare_images(cropped_image, noisy_image, reconstructed_image, f'{org_name}_{str(i)}', float(org_exp_time),
                                                 float(new_exp_time), combined_image_dir, kwargs_source, if_selected)
            if results is not None:
                stats, (flux_rec, flux_org), (flux_error_rec, flux_error_org), (org_df, noisy_df, rec_df) = results 
                stats['epoch'] = epoch
                stats['model'] = model_dir
                
                all_flux_rec.extend(flux_rec)
                all_flux_org.extend(flux_org)
                all_flux_error_rec.extend(flux_error_rec)
                all_flux_error_org.extend(flux_error_org)
                metrics.append(stats)

                if not rec_df.empty:
                    if model_index == 0:
                        org_dfs.append(org_df)
                        noisy_dfs.append(noisy_df)
                    rec_dfs.append(rec_df)
            i+=1

    metrics = pd.DataFrame(metrics)
    all_flux_rec = np.array(all_flux_rec)
    all_flux_org = np.array(all_flux_org)
    all_flux_error_rec = np.array(all_flux_error_rec)
    all_flux_error_org = np.array(all_flux_error_org)

    if len(rec_dfs):
        if model_index == 0:
            org_dfs = pd.concat(org_dfs, ignore_index=True)
            noisy_dfs = pd.concat(noisy_dfs, ignore_index=True)
            org_dfs['epoch'] = epoch
            org_dfs['model'] = model_dir
            noisy_dfs['epoch'] = epoch
            noisy_dfs['model'] = model_dir
        else:
            org_dfs = pd.DataFrame(org_dfs)
            noisy_dfs = pd.DataFrame(noisy_dfs)
        rec_dfs = pd.concat(rec_dfs, ignore_index=True)
        rec_dfs['epoch'] = epoch
        rec_dfs['model'] = model_dir
    else:
        org_dfs = pd.DataFrame(org_dfs)
        noisy_dfs = pd.DataFrame(noisy_dfs)
        rec_dfs = pd.DataFrame(rec_dfs)
    try:
        aggregated_stats = aggregate_df(metrics, all_flux_rec, all_flux_org, all_flux_error_rec, all_flux_error_org)
        aggregated_stats['epoch'] = epoch
        aggregated_stats['model'] = model_dir
        aggregated_metrics.append(aggregated_stats)
    except Exception as err:
        logging.warning(f"{err}")
    return metrics, pd.DataFrame(aggregated_metrics), (org_dfs, noisy_dfs, rec_dfs)

def process_models(job, output_filedir, data_dir, kwargs_source, 
                    workers=8, frac=0.1, parallel=True, patch_size=(256, 256, 1), stride=(128, 128, 1),
                    weighting='gaussian', batch_size=16):
    
    if job is not None and len(job) == 4:
        model_dir, model_files, scales, test_images_df = job

    all_metrics = []
    aggregated_metrics = []
    org_dfs = []
    noisy_dfs = []
    rec_dfs = []
    if parallel:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_single_model_no_mosaic, model_file, model_dir, data_dir, scales,
                                        test_images_df, output_filedir, kwargs_source, patch_size, stride,
                                            weighting, batch_size, frac, index) for index, model_file in enumerate(model_files)]
        for future in as_completed(futures):
            if future is not None:
                try:
                    result = future.result()
                    if result is not None:
                        metrics, aggregated_metric, (org_df, noisy_df, rec_df) = result
                        if not metrics.empty:
                            all_metrics.append(metrics)
                        if not aggregated_metric.empty:
                            aggregated_metrics.append(aggregated_metric)
                        if not org_df.empty:
                            org_dfs.append(org_df)
                            noisy_dfs.append(noisy_df)
                        if not rec_df.empty:
                            rec_dfs.append(rec_df)
                except Exception as err:
                    print(err)
    else:
        for index, model_file in enumerate(model_files):
            result = process_single_model_no_mosaic(model_file, model_dir, data_dir, scales, test_images_df, 
                            output_filedir, kwargs_source, patch_size, stride, weighting, batch_size, frac, index)
            if result is not None:
                metrics, aggregated_metric, (org_df, noisy_df, rec_df) = result
                if not metrics.empty:
                    all_metrics.append(metrics)
                if not aggregated_metric.empty:
                    aggregated_metrics.append(aggregated_metric)
                if not org_df.empty:
                    org_dfs.append(org_df)
                    noisy_dfs.append(noisy_df)
                if not rec_df.empty:
                    rec_dfs.append(rec_df)

    if all_metrics:
        all_metrics = pd.concat(all_metrics, ignore_index=True)
    else:
        all_metrics = pd.DataFrame()
    if aggregated_metrics:
        aggregated_metrics = pd.concat(aggregated_metrics, ignore_index=True)
    else:
        aggregated_metrics = pd.DataFrame()

    if len(org_dfs):
        org_dfs = pd.concat(org_dfs, ignore_index=True)
        noisy_dfs = pd.concat(noisy_dfs, ignore_index=True)
        rec_dfs = pd.concat(rec_dfs, ignore_index=True)
    else:
        org_dfs = pd.DataFrame(org_dfs)
        noisy_dfs = pd.DataFrame(noisy_dfs)
        rec_dfs = pd.DataFrame(rec_dfs)
    return all_metrics, aggregated_metrics, (org_dfs, noisy_dfs, rec_dfs)
        
def process_model_with_kwargs(job, output_filedir, data_dir, kwargs_source, workers, frac, parallel, model_kwargs):
    return process_models(job, output_filedir, data_dir, kwargs_source, workers, frac, parallel, **model_kwargs)
    
def main(models_dir, output_filedir, data_dir, data_kwargs, model_kwargs, kwargs_source, total_workers, max_workers, frac, condition, filter_model, n, parallel=True, parallel_epoch=False):
    try:
        #test_images_df = get_test_images(**data_kwargs)
        test_images_df = pd.read_csv(data_kwargs['metadata_filepath']).sample(n=data_kwargs['N'])
    except Exception as err:
        logging.warning(f"{err}")
    if test_images_df is None:
        return 

    my_dict = find_best_performing_models(models_dir, condition, filter_model, n)
    print(my_dict)

    jobs = []
    for model_dir in my_dict.keys():
        scales = decide_scale(model_dir)
        jobs.append((model_dir, my_dict[model_dir], scales, test_images_df))
    metrics = []
    aggregated_metrics = []
    org_dfs = []
    noisy_dfs = []
    rec_dfs = []

    if parallel:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_model_with_kwargs,
                                        job, output_filedir, data_dir,
                                        kwargs_source,
                                        total_workers//max_workers, frac, parallel_epoch, 
                                        model_kwargs) for job in jobs]
        for future in as_completed(futures):
            if future is not None:
                try:
                    result = future.result()
                    if result is not None:
                        metric, aggregated_metric, (org_df, noisy_df, rec_df) = result
                        if not metric.empty:
                            metrics.append(metric)
                        if not aggregated_metric.empty:
                            aggregated_metrics.append(aggregated_metric)
                        if not org_df.empty and len(org_dfs) == 0:
                            org_dfs.append(org_df)
                            noisy_dfs.append(noisy_df)
                        if not rec_df.empty:
                            rec_dfs.append(rec_df)
                except Exception as err:
                    print(err)
    else:
        for job in jobs:
            result = process_models(job, output_filedir, data_dir, kwargs_source, 
                            workers=total_workers//max_workers, frac=frac,
                             parallel=parallel_epoch, **model_kwargs)
            if result is not None:
                metric, aggregated_metric, (org_df, noisy_df, rec_df) = result
                if not metric.empty:
                    metrics.append(metric)
                if not aggregated_metric.empty:
                    aggregated_metrics.append(aggregated_metric)
                if not org_df.empty and len(org_dfs) == 0:
                    org_dfs.append(org_df)
                    noisy_dfs.append(noisy_df)
                if not rec_df.empty:
                    rec_dfs.append(rec_df)

    if metrics:
        metrics = pd.concat(metrics, ignore_index=True)
    else:
        metrics = pd.DataFrame()

    if aggregated_metrics:
        aggregated_metrics = pd.concat(aggregated_metrics, ignore_index=True)
    else:
        aggregated_metrics = pd.DataFrame()
    metrics.to_csv(os.path.join(output_filedir, 'all_metrics.csv'), index=False)
    aggregated_metrics.to_csv(os.path.join(output_filedir, 'aggregated_metrics.csv'), index=False)

    if len(org_dfs):
        org_dfs = pd.concat(org_dfs, ignore_index=True)
        noisy_dfs = pd.concat(noisy_dfs, ignore_index=True)
        rec_dfs = pd.concat(rec_dfs, ignore_index=True)
    else:
        org_dfs = pd.DataFrame(org_dfs)
        noisy_dfs = pd.DataFrame(noisy_dfs)
        rec_dfs = pd.DataFrame(rec_dfs)
    if not org_dfs.empty:
        org_dfs.to_csv(os.path.join(output_filedir, 'org_catalog.csv'), index=False)
        noisy_dfs.to_csv(os.path.join(output_filedir, 'noisy_catalog.csv'), index=False)
        rec_dfs.to_csv(os.path.join(output_filedir, 'rec_catalog.csv'), index=False)

def condition(f):
    #return True
    #return f in ['model_gpu_scaling_z_scale_random', 'model_gpu_scaling_min_max_random', 
    #             'model_gpu_no_scaling_3_e5_random', 'model_gpu_latest_1e5_dropout_normalized_sampled_log_min_max',
    #            'model_gpu_latest_1e5', 'model_gpu']
    #return f in ['z_scale_1e4_random_gal_data_3000', 'no_scale_1e4_random_gal_data_3000', 'z_scale_1e4_random_3000', 'no_scale_1e4_random_3000', 'no_scale_3e5_random', 'no_scale_3e5_random_gal_data',
    #              'no_scale_1e4_random', 'no_scale_1e4_random_gal_data', 'z_scale_1e4_random_gal_data', 'z_scale_1e4_random', 'model_gpu']
    #return f in ['new_z_scale', 'new_z_scale_gal_data',
    #             'new_no_scale', 'new_no_scale_gal_data', 
    #             'z_scale_1e4_random_3000', 'no_scale_1e4_random_3000',
    #             'z_scale_1e4_random_gal_data_3000', 'no_scale_1e4_random_gal_data_3000',
    #             'model_gpu']
    return f in [
        'new_z_scale', 'new_z_scale_gal_data',
        'new_no_scale', 'new_no_scale_gal_data', 
        'z_scale_gal_data_latest',
        'normal_scale_gal_data_latest'
    ]

def get_top_best_models(file_list, x):
    if not isinstance(x, int) or x <= 0:
        return []

    valid_files = []
    for f in file_list:
        try:
            base = os.path.basename(f)
            num_part = base.split('_')[-1].split('.')[0]
            num = int(num_part)
            valid_files.append((f, num))
        except (ValueError, IndexError):
            continue  # Skip malformed filenames

    valid_files.sort(key=lambda item: item[1], reverse=True)

    # Extract filenames
    top_files = [f[0] for f in valid_files[:x]]

    # Optional: Warn if fewer than x valid files are found
    if len(top_files) < x:
        print(f"Warning: Only {len(top_files)} valid files found, requested {x}.")
    return top_files

def get_model_by_modulo(file_list, modulo=75):
    selected = []
    my_dict = {}
    
    for file in file_list:
        if 'final' in file:
            selected.append(file)
        else:
            basename = os.path.basename(file).split('.')[0]
            parts = basename.split('_')
            for part in parts:
                try:
                    number = int(part)
                    my_dict[number] = file
                    break 
                except Exception:
                    continue

    for index, (epoch, file) in enumerate(sorted(my_dict.items())):
        if epoch % modulo == 0 and epoch <= 550:
            selected.append(file)
        elif index == len(my_dict) - 1:
            selected.append(file)
    return selected
    
if __name__ == '__main__':

    os.chdir(os.path.dirname(__file__))
    output_filedir = "./metrics_updated"
    blend = False
    parallel = True
    parallel_epoch = False
    models_dir = './models'
    data_dir = "./nsigma2_fprint_5_npixels_5"
    data_dir = "./gal_data"
    total_workers = 36
    max_workers = 6
    frac = 0.20
    modulo = 50

    if len(sys.argv) > 1:
        output_filedir = sys.argv[1]
    if len(sys.argv) > 2:
        blend = int(sys.argv[2]) == 1
    if len(sys.argv) > 3:
        parallel = int(sys.argv[3]) == 1

    data_kwargs = {
        "N": 150,
        #"metadata_filepath": f"./{data_dir}/metadata_filepath_enriched_with_noise.csv",
        "metadata_filepath" : 'selected_eval_pictures.csv',
        "metadata_filepath" : 'selected_eval_pictures_gal_data.csv',
        "exp_col": "sci_actual_duration",
        "sigma_col": "light_std",
        "location_col": "location",
        "signal_col": "mean",
        "pep_id_col": "sci_pep_id",
        "dataset_col": "sci_data_set_name",
        "low": 200,
        "high": 20000,
        "factor": 5,
        #"seed" : 42,
        "seed": 26
    }

    model_kwargs = {
        'patch_size' : (256, 256, 1), 
        'stride' : (64, 64, 1), 
        'weighting' : 'average',
        'batch_size' : 256
    }

    kwargs_source = {
        'sigma' : 3,
        'maxiters' : 25, 
        'nsigma' : 3,
        'npixels' : 5,
        'nlevels' : 32,
        'contrast' : 0.001,
        'footprint_radius' : 5,
        'distance_threshold' : 5,
        'deblend' : blend, 
        'deblend_timeout' : 30,
        'alpha' : 1,
        'beta' : 1,
        'gamma' : 1,
        'k1' : 0.01,
        'k2' : 0.03,
        'func' : wrap_extract_sources,
        'thresh': 2.25,  # Threshold for object extraction
        'org_thresh': 2,  # Threshold for object extraction
        'radius_factor': 6.0,  # Factor for Kron radius
        'PHOT_FLUXFRAC': 0.5,  # Fraction for flux radius
        'r_min': 1.75,  # Minimum diameter for circular apertures
        'elongation_fraction': 1.5,  # Elongation threshold for galaxies
        'PHOT_AUTOPARAMS': 2.5,  # Aperture parameter for SEP
        "maskthresh": 0.0,  # float, optional: Threshold for pixel masking
        "minarea": 8,  # int, optional: Minimum number of pixels required for an object
        "org_minarea": 6,  # int, optional: Minimum number of pixels required for an object
        "filter_type": 'matched',  # {'matched', 'conv'}, optional: Filter treatment type
        "deblend_nthresh": 32,  # int, optional: Number of thresholds for deblending
        "deblend_cont": 0.005,  # float, optional: Minimum contrast ratio for deblending
        "clean": True,  # bool, optional: Perform cleaning or not
        "clean_param": 1.0  # float, optional: Cleaning parameter
    }

    main(models_dir, output_filedir, data_dir, data_kwargs, model_kwargs,
     kwargs_source, total_workers, max_workers, frac, condition, get_model_by_modulo, modulo, parallel, parallel_epoch)