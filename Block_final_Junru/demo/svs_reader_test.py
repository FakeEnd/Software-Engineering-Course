"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/guides.html?#readers
"""
import openslide
import numpy as np
from PIL import Image


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
        path = path[0]
    if path.endswith(".svs"):
        # return None
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

    myPyramid = []
    slide = openslide.OpenSlide(path)
    for level in range(slide.level_count):
        dims = slide.level_dimensions[level]
        img = slide.read_region((0, 0), level, dims)
        img_np = np.array(img)[:, :, :3]  # Convert to numpy array and drop alpha channel
        myPyramid.append(img_np)
    
    # Define additional parameters such as contrast limits
    add_kwargs = {'contrast_limits': [0, 255], "multiscale": True}
    layer_type = "image"  # specifies that the layer is an image
    return [(layer_data, add_kwargs, layer_type) for layer_data in myPyramid]
