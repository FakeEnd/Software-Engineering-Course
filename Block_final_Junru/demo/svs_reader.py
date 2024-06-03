import openslide
import numpy as np
from PIL import Image

# Path to your SVS file
svs_path = 'path_to_your_svs_file.svs'

# Open the SVS file using OpenSlide
slide = openslide.OpenSlide(svs_path)

# Prepare the pyramid
myPyramid = []

# Loop through available levels
for level in range(slide.level_count):
    # Get the dimensions of the image at this level
    dims = slide.level_dimensions[level]
    # Read the region of the image at this level
    img = slide.read_region((0, 0), level, dims)
    # Convert the PIL image to a numpy array and drop the alpha channel
    img_np = np.array(img)[:, :, :3]
    # Append to the list
    myPyramid.append(img_np)

# Optionally, you can test this in Napari to see if it works
# import napari
# viewer = napari.Viewer()
# viewer.add_image(myPyramid, contrast_limits=[0, 255])
# napari.run()
