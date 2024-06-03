import tensorflow as tf
from tensorflow.keras.datasets import mnist, cifar10

import numpy as np
import matplotlib.pyplot as plt

class Dataset:
    """
    Base class for an image dataset
    
    Training and test images are stored as attributes (images_train and
    images_test, respectively). Same for the training and test labels
    (labels_train and labels_test, respectively.)
    """
    
    def __init__(self):
        """Load the dataset and perform some preprocessing. 
        """        
        
        (x_train, y_train), (x_test, y_test) = self.load_data()
        
        self.images_train = self.preprocess_images(x_train)
        self.labels_train = y_train
        
        self.images_test = self.preprocess_images(x_test)
        self.labels_test = y_test
    
    def load_data(self):
        # As we can see in __init__, this needs to load the data and return it
        # in the form (x_train, y_train), (x_test, y_test)
        
        raise NotImplementedError('load_data needs to be overridden in your subclass')
        
    def preprocess_images(self, images):
        # This is where images will be preprocessed, such as converting from
        # integer values to float values and resizing.
        
        raise NotImplementedError('preprocess_images needs to be overridden in your subclass')
    
    def show_example_images(self, output_path):
        """Create a figure with some example images.

        Args:
            output_path (str): Path to save the figure.
        """
        # Select 25 images to show
        example_images = self.images_train[:25]
        example_labels = self.labels_train[:25]   
        
        # Use the gray colormap if the image is grayscale (last dimension is 1),
        # but not if the image is RGB color (last dimension is 3).
        if example_images.shape[-1] == 1:
            cmap = 'gray'
        else:
            cmap = None
        
        # Create subplots for each image
        fig, ax = plt.subplots(5, 5, gridspec_kw={'hspace': 0.6}, figsize=(5, 5))
        for i in range(25):
            subfig = ax.flatten()[i]
            subfig.imshow(example_images[i], cmap=cmap)
            subfig.set_title(example_labels[i])
            subfig.axis('off') # Disable the axis ticks
            
        # facecolor='white' disables the transparent background when saving as PNG
        fig.savefig(output_path, facecolor='white')
        fig.show()
        

class MNIST(Dataset):
    """
    Simple class to hold MNIST data. 
        
    The images_train and images_test attributes contain grayscale 2D images with
    floating point-valued pixels, while the labels_train and labels_test
    attributes contain the class labels for each image (which digit it
    represents).
    
    This inherits from the Dataset superclass. We can inherit the __init__ and
    show_example_images methods since they're generic and dataset-agnostic.
    However, we need to override the load_data and preprocess_images methods to
    do operations specific to MNIST.
    
    """  
    
    def load_data(self):
        # Load MNIST data using the built-in Keras function      
        
        # This Keras function downloads MNIST data to your home directory, if
        # it's not already there, then loads it as numpy arrays.
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        # The images are 28 x 28, so the x arrays will have shape n_images x 28
        # x 28. The labels are ordinal and range from 0-9, so the y arrays will
        # have shape n_images x 1.
        return (x_train, y_train), (x_test, y_test)
        
    def preprocess_images(self, images):
        """Rescale image pixel values to float32 (0 to 1) and resize to 32 x 32.

        Args:
            images (array): numpy array of images.

        Returns:
            array: preprocessed images
        """        
        
        # Images are originally in uint8 (integer) format, with pixel values between 0
        # and 255. We need to convert to float32 values between 0 and 1.
        images = images.astype('float32') / 255.

        # Additionally, Tensorflow expects 2D images to have a 3rd dimension
        # corresponding to RGB color channels. Since MNIST is grayscale, we'll just add
        # a length-1 dimension.
        images = np.expand_dims(images, axis=-1)

        # MNIST images are originally 28 x 28. Working with convolutional neural
        # networks is a LOT easier when the image size is a power of 2. These models
        # typically use striding or pooling operations at various layers to reduce the
        # image dimensions by half, so using 28 x 28 images results in weird odd-length
        # dimensions like 7 x 7 later. To avoid that, we'll resample to 32 x 32 now.
        images = tf.image.resize(images, size=(32, 32), method=tf.image.ResizeMethod.BICUBIC)

        # tf.image.resize returns a tensor, so convert back to numpy array
        images = images.numpy()

        # Clip image values to between 0 and 1, since bicubic interpolation may have
        # caused some values to occur outside this range.
        images = np.clip(images, 0.0, 1.0)
        
        return images
    

class CIFAR10(Dataset):
    """
    Simple class to hold CIFAR10 data. 
        
    The images_train and images_test attributes contain color 2D images with
    floating point-valued pixels, while the labels_train and labels_test
    attributes contain the class labels for each image.
    
    This inherits from the Dataset superclass. We can inherit the __init__
    method since it's generic and dataset-agnostic. However, we need to override
    the load_data and preprocess_images methods to do operations specific to
    MNIST. We will also override show_example_images so that it shows the text
    labels for each image instead of just an integer label.
    
    """  
    
    def load_data(self):
        # Load CIFAR10 data using the built-in Keras function      
        
        # This Keras function downloads CIFAR10 data to your home directory, if
        # it's not already there, then loads it as numpy arrays.
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        
        # The images are 32 x 32, so the x arrays will have shape n_images x 32
        # x 32. The labels are ordinal and range from 0-9, so the y arrays will
        # have shape n_images x 1.
        return (x_train, y_train), (x_test, y_test)
    
    def preprocess_images(self, images):
        """Rescale image pixel values to float32 (0 to 1).

        Args:
            images (array): numpy array of images.

        Returns:
            array: preprocessed images
        """        
        
        # Images are originally in uint8 (integer) format, with pixel values between 0
        # and 255. We need to convert to float32 values between 0 and 1.
        images = images.astype('float32') / 255.
        
        return images
    
    def show_example_images(self, output_path):
        """Bonus: Override the original show_example_images methods so that it
        shows the text labels for each CIFAR10 class instead of just the
        integer label (0-9).

        Args:
            output_path (str): Path to save the figure.
        """
        # The 10 class names in CIFAR10
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Select 25 images to show
        example_images = self.images_train[:25]
        example_labels = self.labels_train[:25]   
        
        # Create subplots for each image
        fig, ax = plt.subplots(5, 5, gridspec_kw={'hspace': 0.6}, figsize=(5, 5))
        for i in range(25):
            subfig = ax.flatten()[i]
            subfig.imshow(example_images[i])
            label = example_labels[i, 0]
            subfig.set_title(classes[label])
            subfig.axis('off') # Disable the axis ticks
            
        fig.savefig(output_path, facecolor='white')
        fig.show()
        
        
class LiveCell(Dataset):
    """
    Here's another Dataset subclass for the melanoma live cell imaging dataset.
    Like the other examples, we need to override load_data to do the specific
    loading steps for this dataset and preprocess_images to do the specific
    preprocessing steps. In this case, preprocessing includes cropping the
    center of each image (they have a lot of empty space around the cell) and
    downsampling to 32 x 32 (original images are 256 x 256). This is to reduce
    training time and model size. 

    """
    
    # Path to prepared datasets in numpy npz files
    #train_path = '/archive/bioinformatics/DLLab/KevinNguyen/data/melanoma/allpdx_selecteddates/data_train.npz'
    #test_path = '/archive/bioinformatics/DLLab/KevinNguyen/data/melanoma/allpdx_selecteddates/data_test.npz'
    train_path = '/archive/bioinformatics/DLLab/AlbertMontillo/src/SWE22/data/melanoma/allpdx_selecteddates/data_train.npz' 
    test_path = '/archive/bioinformatics/DLLab/AlbertMontillo/src/SWE22/data/melanoma/allpdx_selecteddates/data_test.npz'
    
    def load_data(self):
        # Load dataset from file
        
        train_data = np.load(self.train_path)       
        x_train = train_data['images']
        y_train = train_data['label']       
    
        test_data = np.load(self.test_path)       
        x_test = test_data['images']
        y_test = test_data['label']
        
        # Remap labels from 0 and 1 to high and low-metastatic efficiency
        y_train = np.array(['low' if y == 1 else 'high' for y in y_train])
        y_test = np.array(['low' if y == 1 else 'high' for y in y_test])
        
        # Shuffle data so that images of the same cell aren't right next to each other
        
        # Set a fixed random seed for this operation
        random_state = np.random.get_state()
        np.random.seed(838)
        
        x_train, y_train = self.shuffle(x_train, y_train)
        x_test, y_test = self.shuffle(x_test, y_test)
        
        # Restore the previous random state
        np.random.set_state(random_state)

        return (x_train, y_train), (x_test, y_test)
    
    def shuffle(self, images, labels):
        """Shuffle data using a fixed random seed.

        Args:
            images (array): n x d x d images
            labels (array): n x 1 labels

        Returns:
            array, array: shuffled images, labels
        """        
        
        idx = np.arange(images.shape[0])
        np.random.seed(208)
        np.random.shuffle(idx)
        return images[idx], labels[idx]
    
    def preprocess_images(self, images):
        """Crop the center 50% of each image and downsample to 32 x 32.

        Args:
            images (array): numpy array of images

        Returns:
            array: cropped and downsampled images
        """        
        
        # Crop out the center 50% of each image
        height = images.shape[1]
        width = images.shape[2]
        crop_height = height // 2
        crop_width = width // 2
        crop_top = height // 4
        crop_left = height // 4
        
        images_cropped = images[:, 
                                crop_top:(crop_top + crop_height), 
                                crop_left:(crop_left+crop_width), 
                                :]
        
        # Resize to 64 x 64 resolution
        images_resized = tf.image.resize(images_cropped, size=(32, 32))

        # tf.image.resize returns a tensor, so convert back to numpy array
        images_resized = images_resized.numpy()

        # Clip image values to between 0 and 1, since bicubic interpolation may have
        # caused some values to occur outside this range.
        images_resized = np.clip(images_resized, 0.0, 1.0)
        
        return images_resized


class AlzheimerBrains(LiveCell):
    """
    Structural MRI of Alzheimer's Disease and healthy patients. These are
    T1-weighted scans that have been skull-stripped, and we have taken 2D
    coronal slices through the hippocampi.
    
    The images are originally 192 x 192 grayscale, and we'll downsample to 32 x
    32.

    """
    
    # Path to prepared datasets in numpy npz files
    # Lab paths
    # train_path = '/archive/bioinformatics/DLLab/AlbertMontillo/src/SWE22/data/ADNI23_sMRI/right_hippocampus_slices_2pctnorm/coronal_MNI-6_numpy/alldata/data_train.npz'
    # test_path = '/archive/bioinformatics/DLLab/AlbertMontillo/src/SWE22/data/ADNI23_sMRI/right_hippocampus_slices_2pctnorm/coronal_MNI-6_numpy/alldata/data_test.npz'

    # Course paths
    train_path = '/archive/course/SWE22/shared/week2/data/ADNI23_sMRI/data_train.npz'
    test_path = '/archive/course/SWE22/shared/week2/data/ADNI23_sMRI/data_test.npz'
    
    def load_data(self):
        # Load dataset from file
        
        train_data = np.load(self.train_path)       
        x_train = train_data['images']
        y_train = train_data['label']       
    
        test_data = np.load(self.test_path)       
        x_test = test_data['images']
        y_test = test_data['label']
        
        # Remap labels from 0 and 1 to high and low-metastatic efficiency
        y_train = np.array(['AD' if y == 1 else 'CN' for y in y_train])
        y_test = np.array(['AD' if y == 1 else 'CN' for y in y_test])
        
        # Shuffle data so that images of the same cell aren't right next to each other
        
        # Set a fixed random seed for this operation
        random_state = np.random.get_state()
        np.random.seed(838)
        
        x_train, y_train = self.shuffle(x_train, y_train)
        x_test, y_test = self.shuffle(x_test, y_test)
        
        # Restore the previous random state
        np.random.set_state(random_state)

        return (x_train, y_train), (x_test, y_test)
    
    def preprocess_images(self, images):
        """Downsample to 32 x 32.

        Args:
            images (array): numpy array of images

        Returns:
            array: cropped and downsampled images
        """        
        
        # Resize to 64 x 64 resolution
        images_resized = tf.image.resize(images, size=(32, 32))

        # tf.image.resize returns a tensor, so convert back to numpy array
        images_resized = images_resized.numpy()

        # Clip image values to between 0 and 1, since bicubic interpolation may have
        # caused some values to occur outside this range.
        images_resized = np.clip(images_resized, 0.0, 1.0)
        
        return images_resized
    

class AlzheimerBrains64(AlzheimerBrains):
    '''
    64 x 64 version of AlzheimerBrains
    '''
    
    def preprocess_images(self, images):
        """Downsample to 64 x 64.

        Args:
            images (array): numpy array of images

        Returns:
            array: cropped and downsampled images
        """        
        
        # Resize to 64 x 64 resolution
        images_resized = tf.image.resize(images, size=(64, 64))

        # tf.image.resize returns a tensor, so convert back to numpy array
        images_resized = images_resized.numpy()

        # Clip image values to between 0 and 1, since bicubic interpolation may have
        # caused some values to occur outside this range.
        images_resized = np.clip(images_resized, 0.0, 1.0)
        
        return images_resized