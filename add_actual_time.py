import os, logging
import pandas as pd
import numpy as np
from astropy.io import fits

data_dirs = ['./gal_data', './nsigma2_fprint_5_npixels_5']

# Change to the correct directory
os.chdir(os.path.dirname(__file__))

main_metadata_filename = 'metadata_filepath_enriched_with_noise.csv'
cropped_metadata_filename = 'cropped_images_stats.csv'

def open_fits(filepath, type_of_image='SCI'):
    """
    USED WITH 'data_augment()' function for images where data is given in electrons

    Opens a .fits file, extracts the specified image data and exposure time, 
    and filters based on the adjusted exposure time.

    Args:
        filepath (str): The path to the .fits file.
        type_of_image (str, optional): The image type to extract (e.g., 'SCI'). 
                                       Defaults to 'SCI'

    Returns:
        tuple or None: Returns `image_data` if 
                       the file meets the criteria, otherwise returns `None`.

    Raises:
        Warning: Logs a warning if an error occurs while reading the .fits file.
    """
    try:
        with fits.open(filepath) as f:
            out = None
            for hdu in f:
                if isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU)) and isinstance(hdu.data, np.ndarray):
                    if out is None and type_of_image.lower() == str(hdu.name).lower():
                        out = hdu.data
                        header = hdu.header
                if out is not None:
                    break
            if out is None:
                return None
            return out, header
    except Exception as err:
        logging.warning(f'Error occurred while reading the .fits file: {err}')
        return None
    

for data_dir in data_dirs:
    # Load metadata files
    main_metadata_df = pd.read_csv(os.path.join(data_dir, main_metadata_filename))
    cropped_metadata_df = pd.read_csv(os.path.join(data_dir, cropped_metadata_filename))

    # Dictionary to store extracted header values
    headers = {'sci_data_set_name': [], 'DELTATIM': [], 'SAMPTIME': []}

    for index, row in main_metadata_df.iterrows():  # <-- Fixed iteration
        dataset = row['sci_data_set_name']
        location = row['location']

        try:
            location = os.path.join(data_dir, location[2:])  # Assuming 'location' is the filename
            result = open_fits(location)

            if result is not None:
                data, header = result
                headers['sci_data_set_name'].append(dataset)
                headers['DELTATIM'].append(header.get('DELTATIM', np.nan))  # Use `.get()` to avoid KeyError
                headers['SAMPTIME'].append(header.get('SAMPTIME', np.nan))

        except Exception as err:
            print(f"Error processing {dataset}: {err}")

    # Convert extracted header data to DataFrame
    headers_df = pd.DataFrame(headers)

    # Merge with existing metadata
    main_metadata_df = pd.merge(main_metadata_df, headers_df, on='sci_data_set_name', how='left')
    cropped_metadata_df = pd.merge(cropped_metadata_df, headers_df, on='sci_data_set_name', how='left')

    # Save new extended metadata
    main_metadata_df.to_csv(os.path.join(data_dir, 'extended_' + main_metadata_filename), index=False)
    cropped_metadata_df.to_csv(os.path.join(data_dir, 'extended_' + cropped_metadata_filename), index=False)
