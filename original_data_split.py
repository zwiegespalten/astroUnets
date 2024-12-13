import os, shutil
import numpy as np
import pandas as pd
from astropy.io import fits
from mast import plot_histogram

def split_up(txt_filepath, image_filepath):
    txt_files = [f'{txt_filepath}/{f}' for f in os.listdir(txt_filepath) if f.endswith('.txt') and 'all' not in f]
    
    for txt_file in txt_files:
        with open(txt_file, 'r') as file:
            filenames = [f for f in file.read().split('\n') if 'hst' in f]
            filenames = [f'{f}_drz.fits' if not f.endswith('_drz.fits') else f for f in filenames]
            
        directories = [f'{image_filepath}/{d}' for d in os.listdir(image_filepath) if os.path.isdir(f'{image_filepath}/{d}')]
        for filename in filenames:
            for directory in directories:
                
                location = f'{directory}/{filename}'
                if os.path.exists(location):
                    
                    directory_name = os.path.basename(txt_file).split('_')[0]
                    dst = f'{txt_filepath}/{directory_name}'
                    if not os.path.exists(dst):
                        os.makedirs(dst, exist_ok=True)
                    shutil.copy(location, dst=f'{dst}/{os.path.basename(filename)}')
                    os.remove(location)

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

directory_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(directory_path)
txt_filepath = './original_dataset'
image_filepath = './data'
filters = ['F555W', 'F606W']
split_up(txt_filepath, image_filepath)
create_histograms(txt_filepath, filters)
