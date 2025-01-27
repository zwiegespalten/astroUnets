import matplotlib.pyplot as plt
from astropy.io import fits
import os
import re
import numpy as np

def find_unique_k(directory):
    """
    Find unique base names (k) in the directory that match the pattern *_drz_*_*_*_*.

    Parameters:
    directory (str): Path to the directory to search.

    Returns:
    list of str: List of unique base names.
    """
    pattern = re.compile(r"(.*_drz_\d+_\d+_\d+_\d+)")
    files = os.listdir(directory)
    matches = {pattern.match(f).group(1) for f in files if pattern.match(f)}
    return list(matches)

def visualize_groups(directory, file, vmin=0):
    """
    Visualize groups of FITS files side by side.

    Parameters:
    directory (str): Path to the directory containing FITS files.
    vmin (float): Minimum value for contrast adjustment. Default is 0.
    """
    pics = [f'{directory}/{file}.fits', f'{directory}/{file}_noise.fits', f'{directory}/{file}_output.fits']

    # Iterate through the groups and display/save the images if all files exist
    if all(os.path.exists(filepath) for filepath in pics):
        # Open and read the FITS files
        images = [fits.open(filepath)[0].data for filepath in pics]
        vmax = max([np.max(image) for image in images])
        vmax = [0.3, 0.3]
        maxes = [np.max(image) for image in images]

        # Plot the images in a single row of subplots
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        titles = ['Original', 'Noise', 'Output']

        for ax, image, title, vmax in zip(axs, images, titles, maxes):
        #for ax, image, title in zip(axs, images, titles):
            if title == 'Output':
                continue
            im = ax.imshow(image, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
            ax.set_title(title)
            ax.axis('off')  # Hide axes for better visualization

        # Save the figure
        save_path = f"{directory}/{file}_combined.png"
        plt.savefig(save_path, dpi=300)
        print(f"Saved combined image: {save_path}")
        plt.close(fig)

directory = f'{os.path.dirname(__file__)}/our_data'
files = find_unique_k(directory)
print(files)
for file in files:
   visualize_groups(directory, file)