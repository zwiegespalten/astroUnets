import os, logging
import urllib.request

directory_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(directory_path)

directory = './original_dataset'
txt_files = [os.path.join(directory, f'{f}') for f in os.listdir(directory) if f.endswith('.txt') and 'all' not in f]
for txt_file in txt_files:
    with open(txt_file,'r') as file:
        files = file.read().split('\n')
    files = [f'{f}_drz_fits' for f in files if f.startswith('hst')]
    file_dir = os.path.join(directory, os.path.basename(txt_file).split('_')[0])
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir, exist_ok=True)

    for file in files:
        url = 'https://hla.stsci.edu/cgi-bin/getdata.cgi?dataset=' + file
        #to have valid url, example: https://hla.stsci.edu/cgi-bin/getdata.cgi?dataset=hst_11426_21_wfc3_uvis_f606w_drz.fits
        file = os.path.join(file_dir, file)
        if not os.path.exists(file):
            try:
                urllib.request.urlretrieve(url, file)
            except Exception as err:
                logging.warning(f'an error occurred while downloading the dataset: {err}')
 
#This code will download all fits images with filters F555W and F606W. 
#Example of valid url https://hla.stsci.edu/cgi-bin/getdata.cgi?dataset=hst_11426_21_wfc3_uvis_f606w_drz.fits 
#To filter used data the dataset.txt can be used
#If image appears more times in dataset.txt it means that image was cut into multiple images

