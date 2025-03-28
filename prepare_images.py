import os, logging
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree
from metrics import scale_image, wrap_extract_sources, modify_keras_model
from uncropped_metrics import crop_image
from new_train import open_fits_2, zscore_normalization, inverse_zscore_normalization

#file_dir = './final_images'
file_dir = './latest_gal_data_550'

data_dir = './gal_data'

#model_dir = './final_images'
model_dir = './models/z_scale_normal_data_latest/checkpoints'

output_dir = './final_images_latest'
all_stats_filepath = 'all_metrics.csv'
aggregates_stats_filepath = 'aggregated_metrics.csv'
metadata_filepath = 'metadata_filepath_enriched_with_noise.csv'
model_filepath = 'model_0350.keras'
epoch = '0350'
model_name = 'z_scale_gal_data_latest'
ratio_initial = 2
ratio_growth = 1.5
ratio_count = 8
low = 60
dataset_name = 'sci_data_set_name'
kwargs_source = {
    'sigma' : 3,
    'maxiters' : 25, 
    'nsigma' : 3,
    'npixels' : 5,
    'nlevels' : 32,
    'contrast' : 0.001,
    'footprint_radius' : 5,
    'distance_threshold' : 5,
    'deblend' : False, 
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

def create_ratios(ratio_initial=2, ratio_growth=1.5, ratio_count=8):
    ratios = []
    i = ratio_initial
    for _ in range(ratio_count):
        ratios.append(i)
        i = int(np.ceil(i * ratio_growth))
    return ratios
    
def zscale_visualization(data):
    if data.shape == (256, 256, 1):
        data = data[:,:, 0]
    data = (data - np.mean(data))/np.std(data)
    return data

def create_image(row, n, ratios, model):

    location = row['location']
    location = os.path.join(data_dir, location[2:])
    sigma = row['light_std']
    image = open_fits_2(location)

    if image is None:
        return None
    
    for index, cropped_image in enumerate(crop_image(image)):
        #if index != n:
        #    continue
        noisy = []
        recs = []
        gammas = []
        org = None

        for ratio in ratios:
            new_sigma = sigma*np.sqrt(ratio-1)
            cropped_image = np.nan_to_num(cropped_image, nan=0.0, posinf=0.0, neginf=0.0)
            cropped_image = np.maximum(cropped_image, 1e-8)
            noisy_image = cropped_image + np.random.normal(loc=0, scale=new_sigma, size=cropped_image.shape)
            min_val = np.min(noisy_image)
            if min_val < 0:
                noisy_image -= min_val

            #result = zscore_normalization(noisy_image)
            #if result is not None:
            #    noisy_image = result[0]
            #    args = result[1:]
            input_image = np.array([np.expand_dims(noisy_image, axis=(-1))])
            rec_image = model.predict(input_image)
            if rec_image is not None:
                rec_image = rec_image[0, :, :, 0]
                #rec_image = inverse_zscore_normalization(rec_image, *args)
                #noisy_image = inverse_zscore_normalization(noisy_image, *args)
                if rec_image is not None:
                    gammas.append(ratio)
                    noisy.append(noisy_image)
                    recs.append(rec_image)
                    org = cropped_image
        yield org, noisy, recs, gammas

def create_composite_plot(org, noisy, recs, gammas, label, output_file):
    dirname = os.path.dirname(output_file)
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)

    fig = plt.figure(figsize=(16, 20))
    gs = gridspec.GridSpec(5, 4, height_ratios=[1, 1, 1, 1, 1]) 
    
    ax0 = plt.subplot(gs[0, :]) 
    ax0.imshow(zscale_visualization(org), cmap='gray', vmin=0, vmax=0.5)
    ax0.set_yticks([])
    ax0.set_xticks([])

    axes = [[plt.subplot(gs[i, j]) for j in range(4)] for i in range(1, 5)]

    i = 0
    print(len(gammas))
    for index, (noisy_image, rec_image, gamma) in enumerate(zip(noisy, recs, gammas)):
        j = index % 4
        axes[i][j].imshow(zscale_visualization(noisy_image), cmap='gray', vmin=0, vmax=0.5)
        axes[i][j].set_title(rf"Noisy Image, $\gamma$={int(gamma)}")
        axes[i+1][j].imshow(zscale_visualization(rec_image), cmap='gray', vmin=0, vmax=0.5)
        axes[i+1][j].set_title(rf"Reconstructed Image, $\gamma$={int(gamma)}")

        axes[i][j].set_xticks([])
        axes[i][j].set_yticks([])
        axes[i+1][j].set_yticks([])
        axes[i+1][j].set_xticks([])

        if j == 3:
            i += 2

    ax0.set_title(f"{label}", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_file, dpi=500)
    #plt.show()
    plt.close()

def plot_source_comparison_sep(original_image, noisy_image, reconstructed_image, 
                               x_org, y_org, x_rec, y_rec, 
                               matched_indices_org, unmatched_indices_org, 
                               matched_indices_rec, unmatched_indices_rec,
                               org_a, org_b, rec_a, rec_b, org_theta, rec_theta):
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

        org_ellipses = []
        for x, y, a, b, theta in zip(x_org[unmatched_indices_org], y_org[unmatched_indices_org], 
                                      org_a[unmatched_indices_org], org_b[unmatched_indices_org], 
                                      org_theta[unmatched_indices_org]):
            e = Ellipse(xy=(x, y), width=6*a, height=6*b, angle=np.degrees(theta), 
                        edgecolor='blue', facecolor='none', linewidth=1.5)
            org_ellipses.append(e)

        for x, y, a, b, theta in zip(x_org[matched_indices_org], y_org[matched_indices_org], 
                                      org_a[matched_indices_org], org_b[matched_indices_org], 
                                      org_theta[matched_indices_org]):
            e = Ellipse(xy=(x, y), width=6*a, height=6*b, angle=np.degrees(theta), 
                        edgecolor='red', facecolor='none', linewidth=1.5)
            org_ellipses.append(e)

        rec_ellipses = []
        for x, y, a, b, theta in zip(x_rec[unmatched_indices_rec], y_rec[unmatched_indices_rec], 
                                      rec_a[unmatched_indices_rec], rec_b[unmatched_indices_rec], 
                                      rec_theta[unmatched_indices_rec]):
            e = Ellipse(xy=(x, y), width=6*a, height=6*b, angle=np.degrees(theta), 
                        edgecolor='green', facecolor='none', linewidth=1.5)
            rec_ellipses.append(e)

        for x, y, a, b, theta in zip(x_rec[matched_indices_rec], y_rec[matched_indices_rec], 
                                      rec_a[matched_indices_rec], rec_b[matched_indices_rec], 
                                      rec_theta[matched_indices_rec]):
            e = Ellipse(xy=(x, y), width=6*a, height=6*b, angle=np.degrees(theta), 
                        edgecolor='red', facecolor='none', linewidth=1.5)
            rec_ellipses.append(e)

        return [org_ellipses,  rec_ellipses]# Return the axis objects

    except Exception as err:
        logging.warning(f"An error occurred while creating combined images: {err}")
        return

def compare_images(image_org, noisy_image, image_reconstructed, kwargs):
    if not isinstance(image_org, np.ndarray) or not isinstance(noisy_image, np.ndarray) or not isinstance(image_reconstructed, np.ndarray):
        return
    if image_org.shape != noisy_image.shape or noisy_image.shape != image_reconstructed.shape or image_org.shape != image_reconstructed.shape:
        return

    try:
        x_org, y_org, flux_org, flux_error_org, org_mask, org_a, org_b, org_theta, org_df = kwargs['func'](image_org, 'original', kwargs)
        x_rec, y_rec, flux_rec, flux_error_rec, rec_mask, rec_a, rec_b, rec_theta, rec_df = kwargs['func'](image_reconstructed, 'reconstructed', kwargs)
        x_noisy, y_noisy, flux_noisy, flux_error_noisy, noisy_mask, noisy_a, noisy_b, noisy_theta, noisy_df = kwargs['func'](noisy_image, 'noisy', kwargs)
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
    
    result  = plot_source_comparison_sep(image_org, noisy_image, image_reconstructed, 
                               x_org, y_org, x_rec, y_rec, 
                               matched_indices_image_org, unmatched_indices_image_org, 
                               matched_indices_image_rec, unmatched_indices_image_rec,
                               org_a, org_b, rec_a, rec_b, org_theta, rec_theta)
    return result

def create_composite_plot_detections(org, noisy, recs, gammas, ellipses, label, output_file):
    dirname = os.path.dirname(output_file)
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
  
    fig = plt.figure(figsize=(16, 20))
    gs = gridspec.GridSpec(5, 4, height_ratios=[1, 1, 1, 1, 1]) 
    org, vmin, vmax = scale_image(org)
    ax0 = plt.subplot(gs[0, :]) 
    ax0.imshow(org, cmap='gray')
    ax0.set_yticks([])
    ax0.set_xticks([])
    org_ellipses = ellipses[0][0]
    for ellipse in org_ellipses:
        ax0.add_patch(ellipse)

    axes = [[plt.subplot(gs[i, j]) for j in range(4)] for i in range(1, 5)]

    i = 0
    for index, (noisy_image, rec_image, gamma, ellipsex) in enumerate(zip(noisy, recs, gammas, ellipses)):

        noisy_image, _, _ = scale_image(noisy_image, vmin=None, vmax=None)
        rec_image, _, _ = scale_image(rec_image, vmin=None, vmax=None)
    
        j = index % 4
        axes[i][j].imshow(noisy_image, cmap='gray')
        axes[i][j].set_title(rf"Noisy Image, $\gamma$={int(gamma)}")
        axes[i+1][j].imshow(rec_image, cmap='gray')
        axes[i+1][j].set_title(rf"Reconstructed Image, $\gamma$={int(gamma)}")

        axes[i][j].set_xticks([])
        axes[i][j].set_yticks([])
        axes[i+1][j].set_yticks([])
        axes[i+1][j].set_xticks([])
        for ellipse in ellipsex[1]:
            axes[i+1][j].add_patch(ellipse)
        if j == 3:
            i += 2

    ax0.set_title(f"{label}", fontsize=16)
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='Matched Sources'),
        Line2D([0], [0], color='blue', lw=2, label='Unmatched Sources (Original)'),
        Line2D([0], [0], color='green', lw=2, label='Unmatched Sources (Reconstructed)')
    ]
    
    plt.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.1, -0.2),
                ncol=3, frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.1, wspace=0.05)
    plt.savefig(output_file, dpi=500)
    #plt.show()
    plt.savefig()

def coordinate_detect_source(org, noisy, recs, gammas, label, kwargs, output_file):
    all_ellipses = []
    new_gammas = []
    for noisy_image, image_reconstructed, gamma in zip(noisy, recs, gammas):
        result = compare_images(org, noisy_image, image_reconstructed, kwargs)
        if result is not None:
            all_ellipses.append(result)
            new_gammas.append(gamma)
    if len(all_ellipses):
        create_composite_plot_detections(org, noisy, recs, new_gammas, all_ellipses, label, output_file)
    
def main():
    all_stats_df = pd.read_csv(os.path.join(file_dir, all_stats_filepath))
    all_stats_sub_df = all_stats_df[(all_stats_df['epoch'] == epoch)  & (all_stats_df['model'] == model_name)]
    ratios = create_ratios(ratio_initial, ratio_growth, ratio_count)

    max_ratio = max(ratios)

    agg_df = all_stats_sub_df.groupby("image_id").agg({
        "F-measure": "mean",
        "org_exp_time": "first" 
    }).reset_index()
    agg_df = agg_df[agg_df['org_exp_time']/20 >= low].sort_values(by=['F-measure']).dropna()

    worst_patch = agg_df.iloc[int(np.random.randint(150, 250))]['image_id']
    best_patch = agg_df.iloc[-int(np.random.randint(150, 250))]['image_id']

    my_set = []
    for patch in [worst_patch, best_patch]:
        my_set.append({'dataset':patch.split('_')[0].upper(), 'n': int(patch.split('_')[-1])})

    my_set = pd.DataFrame(my_set)
    metadata_df = pd.read_csv(os.path.join(data_dir, metadata_filepath))
    #selected_metadata_df = metadata_df[metadata_df[dataset_name].isin(my_set['dataset'])]
    metadata_df = metadata_df[metadata_df['location'].str.contains('eval|test', na=False)].sample(frac=1)
    selected_metadata_df = metadata_df[metadata_df['sci_actual_duration']/20 >= low].sample(n=100)

    model_file = os.path.join(model_dir, model_filepath)
    if not os.path.exists(model_file.replace('.keras', '.h5')):
        modify_keras_model(model_file)
    model = tf.keras.models.load_model(model_file.replace('.keras', '.h5'), compile=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss="mae")

    #for (index, row), label in zip(selected_metadata_df.iterrows(), ['worst_image', 'best_image']):
    for (index, row), label in zip(selected_metadata_df.iterrows(), [f"ID{i+1}" for i in range(len(selected_metadata_df))]):
        target = row['sci_targname']
        #n = int(my_set[my_set['dataset'] == row[dataset_name]]['n'])
        n = 100
        id_ = row[dataset_name]
        combined_label = f'target: {target}, sci_data_set_name: {id_}'
        for index_2, (org, noisy, recs, gammas) in  enumerate(create_image(row, n, ratios, model)):
            if org is not None:
                try:
                    create_composite_plot(org, noisy, recs, gammas, combined_label, os.path.join(output_dir, f'{label}_{index_2}.png')) 
                    coordinate_detect_source(org, noisy, recs, gammas, combined_label, kwargs_source, os.path.join(output_dir, f'{label}_{index_2}_detections.png'))
                except Exception as err:
                    print(err)
os.chdir(os.path.dirname(__file__))
main()
