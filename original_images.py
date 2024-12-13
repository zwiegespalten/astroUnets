import os, logging, shutil
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
from astropy.io import fits
from mast import filter_out_mast, plot_histogram, update_file, download_images

def retrieve_ids(string):
    try:
        parts = string.split('_')
        proposal_id = parts[1]
        part2 = parts[2]
        filter_ = parts[5]
        return proposal_id, part2, filter_
    except Exception as err:
        logging.warning(f'error happened while retrieving id: {err}')
        return None
    
def create_filter(proposal_id, filter_):
    if filter_ == 'f555w':
        return {
        'sci_pep_id': str(proposal_id),
        'sci_spec_1234':'F555W,F555W;*,*;F555W,*;F555W;*',
        'sci_obs_type':'image',
        'sci_aec':'S',
        'sci_instrume':'wfc3'
        }
    elif filter_ == 'f606w':
        return {
        'sci_pep_id': str(proposal_id),
        'sci_spec_1234' : 'F606W,F606W;*,*;F606W,*;F606W;*',
        'sci_obs_type':'image',
        'sci_aec':'S',
        'sci_instrume':'wfc3'
        }
    else:
        return None
    
def process_image_metadata(mission, info):
    proposal_id, part2, filter_ = info
    filter_ = create_filter(proposal_id, filter_)

    if filter_:
        temp_data = filter_out_mast(mission, filter_)
        
        if temp_data is not None and not temp_data.empty:
            if 'sci_data_set_name' in temp_data:
                
                result = temp_data[temp_data['sci_data_set_name'].str.contains(part2, na=False)]
                if result is not None and not result.empty:
                    return result
    return None

def split_training_test_validation(txt_directory, image_directory, metadata_filepath, column):
    if os.path.exists(txt_directory):
        try:
            metadata_file = pd.read_csv(metadata_filepath)
            txt_files = [f'{txt_directory}/{f}' for f in os.listdir(txt_directory) if f.endswith('.txt')]
            for txt_file in txt_files:
                if 'all' in os.path.basename(txt_file):
                    continue
                
                directory_type = os.path.basename(txt_file).split('_')[0]
                directory = f'{txt_directory}/{directory_type}'
                os.makedirs(directory, exist_ok=True)
                with open(txt_file,'r') as file:
                    images = file.read().split('\n')
                    images = [f'{image}.fits' for image in images]

                for image in images:
                    try:
                        prop_id = image.split('_')[1]
                        part2 = image.split('_')[2]
                        row = metadata_file[metadata_file['sci_pep_id'].astype(str) == str(prop_id)]
                        if row is not None and not row.empty:
                            id_ = [val for val in row[column].values if part2 in val][0]
                            image_filepath = [f'{image_directory}/{f}' for f in os.listdir(image_directory) if id_ in f][0]
                            shutil.copy(image_filepath, os.path.join(directory, os.path.basename(image_filepath)))     
                        else:
                            pass
                            #logging.warning(f'{image} was not found in the metadata')
                    except Exception:
                        pass
        except Exception as err:
            logging.warning(f'an error occurred while copying the files: {err}')
    else:
        return 

def main(txt_filepath, output_filepath, coluurl_column, save_dir, max_requests=10000, reset_after=5,
          max_workers=16, download=False, period=60, step=10, mission='hst', hist_kwargs=None):
    data = []
    if os.path.exists(output_filepath):
        data = update_file(output_filepath, column, url_column, save_dir, max_requests, reset_after, max_workers, download, period, step)
    else:
        if os.path.exists(txt_filepath):
            with open(txt_filepath, 'r') as file:
                contents = file.read().split('\n')

        image_info = []
        for content in contents:
            result = retrieve_ids(content)
            if result:
                image_info.append(result)

        if image_info:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_image_metadata, mission,info)
                            for info in image_info]
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result is not None and not result.empty:
                            data.append(result)
                    except Exception as err:
                        logging.warning(f'an error occurred in one of threads: {err}')

        data = pd.concat(data)
        data.drop_duplicates(subset=[column], inplace=True)
        data.sort_values(by=['sci_actual_duration'])
        data.reset_index(drop=True, inplace=True)
        data[url_column] = [None] * len(data)
        if not data.empty:
            data.to_csv(output_filepath, index=False)
            plot_histogram(data['sci_actual_duration'], **hist_kwargs)

        data = update_file(output_filepath, column, url_column, save_dir, max_requests, reset_after, max_workers, download, period, step)

    original_images_dir = f'{save_dir}/original_images'
    if download:
        urls = data['URL'].values
        ids = data[column].values
        asyncio.run(download_images(ids, urls, save_dir=original_images_dir, max_requests=max_requests, reset_after=reset_after))

    split_training_test_validation(save_dir , original_images_dir, output_filepath, column)

    return data

def create_metadata_file(filepaths, output_filepath, save=False):
    headers = []
    columns = []
    for filename in filepaths:
        with fits.open(filename) as hdul:
            header = hdul[0].header
            headers.append(header)
            columns.extend([column for column in header])

    columns = set(columns)

    data = {}
    for header in headers:
        for column in columns:
            if not column in data:
                data[column] = []
            if column in header:
                data[column].append(header[column])
            else:
                data[column].append(np.nan)

    data = pd.DataFrame(data)
    if save:
        data.to_csv(output_filepath, index=False)
    return data

def create_histograms(txt_filepath, filters):
    folders = [f'{txt_filepath}/{f}' for f in os.listdir(txt_filepath) if f.endswith('.txt') and 'all' not in f]
    folders =  [('/').join(f.split('/')[:-1]) + '/' + os.path.basename(f).split('_')[0] for f in folders]
    all_filenames = []
    suptitle = ''
    for filter_ in filters:
        filenames = []
        suptitle += filter_ + ', '
        for folder in folders:
            filenames.extend([f'{folder}/{f}' for f in os.listdir(folder) if f.endswith('.fits') and filter_.lower() in f.lower()])

        all_filenames.extend(filenames)
        data = create_metadata_file(filenames, '', save=False)
        plot_histogram(data['EXPTIME'], output_filename=f'{txt_filepath}/hist_org_{filter_}.png', title=f'Histogram of Images with {filter_}')
    
    data = create_metadata_file(all_filenames, '', save=False)
    plot_histogram(data['EXPTIME'], output_filename=f'{txt_filepath}/hist_org.png', title=f'Histogram of Images with {suptitle}')
    

save_dir = './original_dataset'
txt_filepath = f'{save_dir}/all_dataset.txt'
output_filepath = f'{save_dir}/metadata_original_images.csv'
column = 'sci_data_set_name'
url_column = 'URL'
max_requests = 10000
reset_after = 10
max_workers = 32
download = False
period = 60
step = 10
mission = 'hst'
hist_kwargs = {
    'bins' : 20,
    'title' : 'F555W and F606W Images',
    'output_filename' : f'{save_dir}/hist_original_images.png',
    'loc' : 'upper right'
}
directory_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(directory_path)

#main(txt_filepath, output_filepath, column, url_column, save_dir, max_requests, reset_after,
#          max_workers, download, period, step, mission, hist_kwargs)

#image_filepath = './data'
#filters = ['F555W', 'F606W']
#create_histograms(save_dir, filters)

output_filepath = './hst_wfc3_f160W_metadata.csv'
update_file(output_filepath, column, url_column, save_dir, max_requests, reset_after, max_workers, download, period, step)
    

    