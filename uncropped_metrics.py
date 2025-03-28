import os, logging, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from metrics import modify_keras_model, compare_images, wrap_extract_sources, decide_scale
from new_train import open_fits_2
from new_train import min_max_normalization, inverse_min_max_normalization, adaptive_log_transform_and_normalize, inverse_adaptive_log_transform_and_denormalize, zscore_normalization, inverse_zscore_normalization


def turn_exp_to_noisy_sigma(exp, sigma, ratio):
    new_exp = exp / ratio
    new_sigma = np.sqrt(exp / new_exp) * sigma
    return new_sigma, new_exp

def process_data(data_dir, metadata_filename, last_name='FABER', low=200, ratio_initial=2, ratio_growth=1.5, ratio_count=10, output_filename='processed_data.csv'):
    df = pd.read_csv(os.path.join(data_dir, metadata_filename))
    selected_df = df[df['sci_pi_last_name'] == last_name]
    
    ratios = []
    i = ratio_initial
    for _ in range(ratio_count):
        ratios.append(i)
        i = int(np.ceil(i * ratio_growth))

    data = []
    for index, row in selected_df.iterrows():
    #for index, row in df.iterrows():
        location = row['location']
        sci_data_set_name = row['sci_data_set_name']
        exp = row['sci_actual_duration']
        sigma = row['light_std']
    
        for ratio in ratios:
            new_sigma, new_exp = turn_exp_to_noisy_sigma(exp, sigma, ratio)
            noisy_sigma = np.sqrt(new_sigma**2 - sigma**2) if new_sigma**2 > sigma**2 else np.nan
            data.append({
                'sci_data_set_name': sci_data_set_name,
                'sci_actual_duration': exp,
                'location': location,
                'exp_time': exp,
                'new_exp_time': new_exp,
                'combined_sigma': new_sigma,
                'noisy_sigma': noisy_sigma,
                'sigma': sigma,
                'exp_ratio' : ratio
            })

    data_df = pd.DataFrame(data)
    filtered_data = data_df[data_df['new_exp_time'] >= low]
    filtered_data.to_csv(output_filename, index=False)
    return filtered_data

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

def process_subdf(sub_df, model_filepath, data_dir, output_dir, kwargs, bins, scale_on):
    all_hists = {exp_ratio:[np.zeros(len(bins)-1), np.zeros(len(bins)-1), np.zeros(len(bins)-1)] for exp_ratio in sub_df['exp_ratio'].unique()}
    results_df = []
    try:
        if not os.path.exists(model_filepath.replace('.keras', '.h5')):
            modify_keras_model(model_filepath)
        model = tf.keras.models.load_model(model_filepath.replace('.keras', '.h5'), compile=False)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        model.compile(optimizer=optimizer, loss="mae")

    except Exception as err:
        print(err)
        return None

    scaler = descaler = None
    if scale_on:
        scale = decide_scale(model_filepath)
        if scale is None:
            scale_on = False
        else:
            scaler, descaler = scale

    org_dfs = []
    rec_dfs = []
    noisy_dfs = []
    org_set = set()
    for index, row in sub_df.iterrows():
        try:
            location = row.location
            location = os.path.join(data_dir, location[2:])
        except Exception as err:
            print(f"Error with location: {err}")
            continue
        add_org_set = add_noisy_set = False
        noisy_sigma  = row.noisy_sigma
        exp_time = row.sci_actual_duration
        new_exp_time = row.new_exp_time
        image_id = row.sci_data_set_name
        exp_ratio = row['exp_ratio']
        image = open_fits_2(location)
        if image is None:
            continue

        i = 0
        for cropped_image in crop_image(image):
            cropped_image = np.maximum(cropped_image, 1e-8)
            cropped_image = np.nan_to_num(cropped_image, nan=0.0, posinf=0.0, neginf=0.0)
            noisy_image = cropped_image + np.random.normal(loc=0, scale=noisy_sigma, size=cropped_image.shape)
            noisy_image = np.nan_to_num(noisy_image, posinf=0.0, neginf=0.0)
            min_noisy = np.min(noisy_image)
            if min_noisy < 0:
                noisy_image -= min_noisy

            if scale_on:
                args = scaler(noisy_image)
                if args is not None:
                    noisy_image = args[0]
                    func_args = args[1:]
                else:
                    continue
            
            input_image = np.array([np.expand_dims(noisy_image, axis=(-1))])
            rec_image = model.predict(input_image)
            rec_image = rec_image[0, :, :, 0]

            if scale_on:
                rec_image = descaler(rec_image, *func_args)
                noisy_image = descaler(noisy_image, *func_args)
                if rec_image is None or noisy_image is None:
                    continue
                noisy_image = np.nan_to_num(noisy_image, nan=0.0, posinf=0.0, neginf=0.0)
        
            rec_image = np.nan_to_num(rec_image, nan=0.0, posinf=0.0, neginf=0.0)
            rec_min = np.min(rec_image)
            if rec_min < 0:
                rec_image -= rec_min

            all_hists[exp_ratio][0] = np.histogram(cropped_image.flatten(), bins=bins)[0]
            all_hists[exp_ratio][1] = np.histogram(noisy_image.flatten(), bins=bins)[0]
            all_hists[exp_ratio][2] = np.histogram(rec_image.flatten(), bins=bins)[0]
            result = compare_images(cropped_image, noisy_image, rec_image, f'{image_id}_{str(i)}', exp_time, new_exp_time, output_dir, kwargs, True)
            if result is not None:
                stats, (flux_rec, flux_org), (flux_error_rec, flux_error_org), (org_df, noisy_df, rec_df) = result 
                stats['flux_rec'] = flux_rec
                stats['flux_org'] = flux_org
                stats['flux_error_rec'] = flux_error_rec
                stats['flux_error_org'] = flux_error_org
                results_df.append(stats)

                if not org_df.empty:
                    if f'{image_id}_{str(i)}' not in org_set:
                        org_dfs.append(org_df)
                        org_set.add(f'{image_id}_{str(i)}')
                    noisy_dfs.append(noisy_df)
                    rec_dfs.append(rec_df)
            i += 1
    if len(org_dfs):
        org_dfs = pd.concat(org_dfs, ignore_index=False)
        noisy_dfs = pd.concat(noisy_dfs, ignore_index=False)
        rec_dfs = pd.concat(rec_dfs, ignore_index=False)
    else:
        org_dfs = pd.DataFrame(org_dfs)
        noisy_dfs = pd.DataFrame(noisy_dfs)
        rec_dfs = pd.DataFrame(rec_dfs)
    return results_df, all_hists, (org_dfs, noisy_dfs, rec_dfs)

def log_range(min_exp, max_exp):
    data = []
    for i in range(min_exp, max_exp+1):
        for j in range(1,10):
            data.append(j*10.**i)
    data.sort()
    return np.array(data)

def main(N, model_filepath, metadata_filename, data_dir, output_dir, kwargs, workers=4, scale_on=True, min_exp=-5, max_exp=5):
    gal_df = pd.read_csv(metadata_filename)
    gal_df = gal_df.sample(n=min(len(gal_df),N))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    chunk_size = len(gal_df) // workers
    sub_dfs = [gal_df.iloc[i:i + chunk_size] for i in range(0, len(gal_df), chunk_size)]
    bins = log_range(min_exp, max_exp)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_subdf, sub_df, model_filepath, data_dir, output_dir, kwargs, bins, scale_on) for sub_df in sub_dfs]

    results = []
    all_hists = []
    org_dfs = []
    noisy_dfs = []
    rec_dfs = []
    org_set = set()
    for future in as_completed(futures):
        try:
            result = future.result()
            if result is not None:
                data, hists, (org_df, noise_df, rec_df) = result
                results.extend(data)
                all_hists.append(hists)

                if not org_df.empty:
                    temp = org_df[~org_df['image_id'].isin(org_set)]
                    org_set.update(temp['image_id'].tolist())
                    org_dfs.append(temp)
                    noisy_dfs.append(noise_df)
                    rec_dfs.append(rec_df)
        except Exception as err:
            print(err)

    results = pd.DataFrame(results)
    results.to_csv(os.path.join(output_dir, 'results_galdata.csv'), index=False)

    if len(org_dfs):
        org_dfs = pd.concat(org_dfs, ignore_index=False)
        noisy_dfs = pd.concat(noisy_dfs, ignore_index=False)
        rec_dfs = pd.concat(rec_dfs, ignore_index=False)
        org_dfs.to_csv(os.path.join(output_dir, 'org_catalog.csv'), index=False)
        noisy_dfs.to_csv(os.path.join(output_dir, 'noise_catalog.csv'), index=False)
        rec_dfs.to_csv(os.path.join(output_dir, 'rec_catalog.csv'), index=False)
    
    final_hists = {exp_ratio: [np.zeros(len(bins) - 1), np.zeros(len(bins) - 1), np.zeros(len(bins) - 1)]
                for exp_ratio in gal_df['exp_ratio'].unique()}
    for hist_group in all_hists:
        for exp_ratio, hists in hist_group.items():
            final_hists[exp_ratio][0] += hists[0]  # Original
            final_hists[exp_ratio][1] += hists[1]  # Noisy
            final_hists[exp_ratio][2] += hists[2]  # Reconstructed

    hist_data = []
    bin_edges = bins[1:] 
    colors = {
        'original': '#1f77b4',  # A softer blue (from seaborn palette)
        'noisy': '#ff7f0e',  # A warm orange
        'reconstructed': '#2ca02c'  # A fresh green
    }
    for exp_ratio, hists in final_hists.items():
        plt.figure(figsize=(8, 5)) 
        for hist, label in zip(hists, ['original', 'noisy', 'reconstructed']):
            plt.bar(
            bins[:-1], 
            hist, 
            width=np.diff(bins),  
            align='edge', 
            alpha=0.5, 
            label=label, 
            color=colors[label]
            )

            lower_bound = bins[0]
            for val, upper_bound in zip(hist, bin_edges):
                hist_data.append({
                    'label': label,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'exp_ratio': exp_ratio,
                    'count': val
                })
                lower_bound = upper_bound 

        plt.xlabel("Magnitude")
        plt.ylabel("Count")
        plt.xscale("log") 
        plt.title(f"Histogram for Exposure Ratio {int(exp_ratio)}")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'hist_{int(exp_ratio)}.png'), dpi=300) 
        plt.close()

    hist_data = pd.DataFrame(hist_data)
    hist_data.to_csv(os.path.join(output_dir, 'hist_data.csv'), index=False)

if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
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
        'thresh': 2.25,
        'org_thres': 2.0,   # Threshold for object extraction
        'radius_factor': 6.0,  # Factor for Kron radius
        'PHOT_FLUXFRAC': 0.5,  # Fraction for flux radius
        'r_min': 1.75,  # Minimum diameter for circular apertures
        'elongation_fraction': 1.5,  # Elongation threshold for galaxies
        'PHOT_AUTOPARAMS': 2.5,  # Aperture parameter for SEP
        "maskthresh": 0.0,  # float, optional: Threshold for pixel masking
        "minarea": 8,
        "org_minarea" : 6,   # int, optional: Minimum number of pixels required for an object
        "filter_type": 'matched',  # {'matched', 'conv'}, optional: Filter treatment type
        "deblend_nthresh": 32,  # int, optional: Number of thresholds for deblending
        "deblend_cont": 0.005,  # float, optional: Minimum contrast ratio for deblending
        "clean": True,  # bool, optional: Perform cleaning or not
        "clean_param": 1.0  # float, optional: Cleaning parameter
    }

    on_off_flag = False
    on_off = sys.argv[1]
    if on_off == '1':
        on_off_flag = True
    model_name = sys.argv[2]
    model_file_name = sys.argv[3]
    tags = sys.argv[4]

    data_dir = './gal_data'
    #data_dir = './nsigma2_fprint_5_npixels_5'
    metadata_filename = 'selected_eval_pictures_gal_data.csv'
    #metadata_filename = 'selected_eval_pictures.csv'
    model_filepath = f'./models/{model_name}/checkpoints/{model_file_name}.keras'
    
    if not os.path.exists(model_filepath):
        for x in reversed(range(325, 1000, 25)):  # `list()` is unnecessary in Python 3
            digit = f"{x:04d}"  # Ensures zero padding (e.g., "0325", "0350")
            model_file_name = f"model_{digit}"
            model_filepath = f"./models/{model_name}/checkpoints/{model_file_name}.keras"
            if os.path.exists(model_filepath):
                break  # Exit loop if a valid file is found
        else:
            model_filepath = None  # Ensure it's None if no file is found

    output_dir = f"./{'on' if on_off_flag else 'off'}_{model_name}_{tags}"
    N = 20000

    output_filename = './augmented_faber_data.csv'
    filtered_data = process_data(
        data_dir=data_dir,
        metadata_filename='metadata_filepath_enriched_with_noise.csv',
        last_name='FABER',
        low=60,
        ratio_initial=2,
        ratio_growth=1.5,
        ratio_count=10,
        output_filename='./augmented_faber_data.csv'
    )
    
    main(N, model_filepath, output_filename, data_dir, output_dir, kwargs_source, workers=16, scale_on=on_off_flag, min_exp=-5, max_exp=5)
