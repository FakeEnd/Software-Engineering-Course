"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/guides.html?#readers
"""
import openslide
from openslide.deepzoom import DeepZoomGenerator
import dask.array as da
from dask import delayed  # Corrected import for delayed
import numpy as np



def napari_get_reader(path):
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    if isinstance(path, list):
        # reader plugins may be handed single path, or a list of paths.
        # if it is a list, it is assumed to be an image stack...
        # so we are only going to look at the first file.
        path = path[0]

    # if we know we cannot read the file, we immediately return None.
    if not path.endswith(".svs"):
        return None

    # otherwise we return the *function* that can read ``path``.
    return reader_function


def reader_function(path):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of
        layer. Both "meta", and "layer_type" are optional. napari will
        default to layer_type=="image" if not provided
    """
    # Convert path to list if it is a single path string
    if isinstance(path, str):
        path = [path]

    # Initialize an empty list to store dask arrays for each file
    pyramid_layers = []

    for file_path in path:
        slide = openslide.OpenSlide(file_path)
        dz = DeepZoomGenerator(slide, tile_size=254, overlap=1, limit_bounds=False)
        
        @delayed(pure=True)
        def get_tile(level, column, row):
            tile = dz.get_tile(level, (column, row))
            return np.array(tile).transpose((1, 0, 2))
    
        layers = []
        # Reverse to ensure smallest to largest
        for level in range(dz.level_count - 1, -1, -1):  
            level_tiles = dz.level_tiles[level]
            sample_tile_shape = get_tile(level, 0, 0).shape.compute()
            n_tiles_x, n_tiles_y = level_tiles[0], level_tiles[1]
            
            level_dimensions = dz.level_dimensions[level]
            if n_tiles_x <= 1 or n_tiles_y <= 1:
                print(
                    f"Ignoring Level {level} with dimensions: {level_dimensions}"
                )
                continue
            else:
                print(f"Reading Level {level} with dimensions: {level_dimensions}")
            
            rows = range(n_tiles_y - 1)
            cols = range(n_tiles_x - 1)
            # Convert list of delayed operations to dask array only once
            layer = da.concatenate(
                [
                    da.concatenate(
                        [
                            da.from_delayed(
                                get_tile(level, col, row),
                                sample_tile_shape,
                                np.uint8,
                            )
                            for row in rows
                        ],
                        allow_unknown_chunksizes=False,
                        axis=1,
                    )
                    for col in cols
                ],
                allow_unknown_chunksizes=False,
            )
            layers.append(layer)
            
        pyramid_layers.append(layers)

    # Since reader functions expect a list of tuples
    # return [(layer, {'name': f'Level {i}', "multiscale": True, 'contrast_limits': [0, 255]}, 'image') for i, layer in enumerate(pyramid_layers)]
    return [(layer, { "multiscale": True, 'contrast_limits': [0, 255]}, 'image') for i, layer in enumerate(pyramid_layers)]
