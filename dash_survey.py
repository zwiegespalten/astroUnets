import os, logging
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from new_train import save_fits, open_fits_2, zscore_normalization, inverse_zscore_normalization
from metrics import modify_keras_model, scale_image
from uncropped_metrics import crop_image

os.chdir(os.path.dirname(__file__))
file_dir = './nsigma2_fprint_5_npixels_5'
metadata_filename = 'metadata_filepath_enriched_with_noise.csv'
model_dir = './models/z_scale_normal_data_latest/checkpoints'
model_filepath = 'model_0350.keras'
output_dir = 'rec_dash_images'

def create_image(row, model, output_dir, file_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    dataset_name = row['sci_data_set_name']
    location = row['location']
    exp_time = row['sci_actual_duration']
    location = os.path.join(file_dir, location[2:])
    image = open_fits_2(location)

    if image is None:
        return None
    for index, cropped_image in enumerate(crop_image(image)):
        try:
            cropped_image = np.nan_to_num(cropped_image, nan=0.0, posinf=0.0, neginf=0.0)
            cropped_image = np.maximum(cropped_image, 1e-8)
            min_val = np.min(cropped_image)
            if min_val < 0:
                cropped_image -= min_val

            result = zscore_normalization(cropped_image)
            if result is not None:
                cropped_image = result[0]
                args = result[1:]
            input_image = np.array([np.expand_dims(cropped_image, axis=(-1))])
            rec_image = model.predict(input_image)
            if rec_image is not None:
                rec_image = rec_image[0, :, :, 0]
                rec_image = inverse_zscore_normalization(rec_image, *args)
                if rec_image is not None:
                    min_val = np.min(rec_image)
                    if min_val < 0:
                        rec_image -= min_val
                    rec_image = np.nan_to_num(rec_image, nan=0.0, posinf=0.0, neginf=0.0)
                    try:
                        save_fits(rec_image, f'{str(round(exp_time))}_' + dataset_name + f"_rec_{index}", output_dir + '/')
                    except Exception as err:
                        logging.warning(err)

                    rec_image_scaled = zscale_visualization(rec_image)
                    cropped_image_scaled = zscale_visualization(cropped_image)

                    try:
                        if rec_image_scaled is not None and cropped_image_scaled is not None:
                            fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns
                            
                            axes[0].imshow(cropped_image_scaled, cmap='gray', vmin=0, vmax=0.5)
                            axes[0].set_title("Original Image")
                            axes[0].axis("off")

                            axes[1].imshow(rec_image_scaled, cmap='gray', vmin=0, vmax=0.5)  # You can replace this with another image
                            axes[1].set_title("Model Output")
                            axes[1].axis("off")

                            plt.tight_layout()
                            plt.savefig(os.path.join(output_dir, f'{str(round(exp_time))}_' + dataset_name + f"_rec_{index}.png"), dpi=300)
                            plt.close()
                    except Exception as err:
                        logging.warning(err)

                    try:
                        result = scale_image(cropped_image)
                        if result is not None:
                            cropped_image, _, _ = result
                        result = scale_image(rec_image)
                        if result is not None:
                            rec_image, _, _ = result

                        if rec_image is not None and cropped_image is not None:
                            fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns
                            
                            axes[0].imshow(cropped_image, cmap='gray')
                            axes[0].set_title("Original Image")
                            axes[0].axis("off")

                            axes[1].imshow(rec_image, cmap='gray')  # You can replace this with another image
                            axes[1].set_title("Model Output")
                            axes[1].axis("off")

                            plt.tight_layout()
                            plt.savefig(os.path.join(output_dir, f'{str(round(exp_time))}_' + dataset_name + f"_ZInterval_rec_{index}.png"), dpi=300)
                            plt.close()
                    except Exception as err:
                        logging.warning(err)

        except Exception as err:
            logging.warning(f'error at predicting {err}')
            continue

def zscale_visualization(data):
    if data.shape == (256, 256, 1):
        data = data[:,:, 0]
    data = (data - np.mean(data))/np.std(data)
    return data

def main(file_dir, metadata_filename, model_dir, model_filepath, output_dir):
    metadata_df = pd.read_csv(os.path.join(file_dir, metadata_filename))
    selected_metadata_df = metadata_df[metadata_df['sci_pi_last_name'] == 'MOMCHEVA'].reset_index(drop=True)

    model_file = os.path.join(model_dir, model_filepath)
    if not os.path.exists(model_file.replace('.keras', '.h5')):
        modify_keras_model(model_file)
    model = tf.keras.models.load_model(model_file.replace('.keras', '.h5'), compile=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    model.compile(optimizer=optimizer, loss="mae")

    for index, row in selected_metadata_df.iterrows():
        try:
            create_image(row, model, output_dir, file_dir)
        except Exception as err:
            logging.warning(err)

main(file_dir, metadata_filename, model_dir, model_filepath, output_dir)