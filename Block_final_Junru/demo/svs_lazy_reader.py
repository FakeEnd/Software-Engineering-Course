import openslide
from openslide.deepzoom import DeepZoomGenerator
import dask.array as da
import numpy as np
import napari

def load_svs_with_dask(path):
    slide = openslide.open_slide(path)
    dz = DeepZoomGenerator(slide, tile_size=254, overlap=1, limit_bounds=False)  # adjust tile_size & overlap as needed

    def get_tile(dz, level, loc):
        return np.array(dz.get_tile(level, loc))

    layers = []
    for level in range(dz.level_count):
        level_tiles = dz.level_tiles[level]
        lazy_tiles = [
            da.from_delayed(
                da.delayed(get_tile)(dz, level, (x, y)),
                shape=(254, 254, 3),  # this should match your tile size and channels
                dtype=np.uint8
            )
            for y in range(level_tiles[1])
            for x in range(level_tiles[0])
        ]
        layer = da.block(lazy_tiles).reshape((level_tiles[1]*254, level_tiles[0]*254, 3))  # adjust the reshape according to the actual tile size and grid
        layers.append(layer)

    # Ensure layers are in the correct order (smallest to largest)
    layers = layers[::-1]
    return layers

# Now use Napari to display the dask array
myPyramid = load_svs_with_dask('path_to_your_svs_file.svs')
