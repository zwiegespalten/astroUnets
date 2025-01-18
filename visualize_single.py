from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

# Function to visualize a single FITS file
def get_dim(file_path):
    try:
        # Open the FITS file
        with fits.open(file_path) as hdul:
            data = hdul[3].data
            return data.shape
    except Exception as e:
        print(f"Error: {e}")


# Function to visualize a single FITS file
def visualize_fits(file_path, filepath2):
    try:
        # Open the FITS file
        with fits.open(file_path) as hdul:
            # Display header information
            print("Header Information:\n")
            print(hdul[0].header)

            # Get the data
            data = hdul[0].data

            # Check if data is not None and is 2D
            if data is None:
                raise ValueError("No data found in the FITS file.")
            
            #elif data.ndim != 2:
            #    raise ValueError("Visualizer supports only 2D FITS data.")

            # Normalize the data for better visualization
            data_normalized = (data - np.min(data)) / (np.max(data) - np.min(data))

            # Plot the data
            plt.figure(figsize=(8, 8))
            plt.imshow(data_normalized, cmap='gray', origin='lower')
            plt.colorbar(label='Intensity')
            plt.title("FITS File Visualization")
            plt.xlabel("X Pixel")
            plt.ylabel("Y Pixel")
            plt.show()
    except Exception as e:
        print(f"Error: {e}")

# Path to the FITS file
file_path2 = 'D:/astro_images/physik_thesis/scripts/astroUnets/our_data/training/originals/ibya10020_drz.fits'
file_path = 'D:/astro_images/physik_thesis/scripts/astroUnets/our_data/training/masked_images/masked_ibya10020_drz.fits'
# Visualize the FITS file
visualize_fits(file_path, file_path2)