import os
import random
import shutil

directory = os.path.dirname(__file__)
os.chdir(directory)

data_directory = './data'
destination = './original_data'
destinations = [os.path.join(destination, d) for d in ['training', 'test', 'eval']]

for dest in destinations:
    os.makedirs(dest, exist_ok=True)

image_filepaths = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.endswith('.fits')]
random.shuffle(image_filepaths)

ratios = (60, 80, 100)  # Ratios in percentage
total_images = len(image_filepaths)
indices = [
    (0, total_images * ratios[0] // 100),
    (total_images * ratios[0] // 100, total_images * ratios[1] // 100),
    (total_images * ratios[1] // 100, total_images * ratios[2] // 100),
]

for dest, (start, end) in zip(destinations, indices):
    images = image_filepaths[start:end]
    for image in images:
        shutil.copy(image, os.path.join(dest, os.path.basename(image)))