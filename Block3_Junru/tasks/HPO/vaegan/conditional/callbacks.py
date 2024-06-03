import os
from matplotlib import pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.keras.callbacks import Callback

# local def. AAM
def sfn(dist_mean, dist_logvar):
    z = tf.random.normal(shape=tf.shape(dist_mean))
    sampledZ = dist_mean + tf.exp(0.5 * dist_logvar) * z
    return sampledZ
    
class GenerateImagesConditional(Callback):
    def __init__(self, output_dir, model, example_labels,
                 cmap='gray',
                 n_generated_images=10, 
                 n_latent_dims=8,
                 class_names=None):
        """Callback for saving examples of synthetic 
        images from a conditional GAN object after each epoch.

        Args:
            output_dir (str): Path to save location. 
            model: Model object.
            example_labels (array): Numpy array containing one-hot labels.
            cmap (str): colormap. Defaults to 'gray'.
            n_generated_images (int, optional): Number of synthetic images 
                to generate. Defaults to 10.
            n_latent_dims (int, optional): Size of latent representation.
                Defaults to 8.
            class_names (list, optional): Name of each class to use in figures.
        """        
        self.output_dir = output_dir
        self.cmap = cmap
        self.n_generated_images = n_generated_images
        self.n_latent_dims = n_latent_dims
        self.model = model
        self.example_labels = example_labels
        self.n_classes = example_labels.shape[1]
        self.class_names = class_names
        
    def on_epoch_end(self, epoch, logs=None):
        """Overrides the on_epoch_end method of the superclass Callback. Here,
        we define what operations should be done, as the name implies, at the
        end of each epoch. This includes saving example reconstructions and
        generating some de novo images.
        
        When overriding a superclass method, it's best to match the signature 
        (i.e. the set of arguments taken) in your new method. In this case, 
        Keras will be calling this method automatically during model training, 
        and it assumes that the method takes these exact two arguments (epoch 
        and logs).

        Args:
            epoch (int): Current epoch number. 
            logs (dict, optional): A dict of metrics from the current epoch. 
                We don't need it here, but since the original Callback.on_epoch_end 
                includes this argument, we have to include it in our overriding 
                method. Defaults to None.
        """        
        del logs # Unused
                                        
        # Generate some synthetic images from random latent representations for
        # each class
        fig, ax = plt.subplots(self.n_generated_images, self.n_classes, 
                               figsize=(self.n_classes, self.n_generated_images))
        
        for i_class in range(self.n_classes):
            labels = np.zeros((self.n_generated_images, self.n_classes))
            labels[:, i_class] = 1
            labels = tf.convert_to_tensor(labels, dtype=tf.float32)
                            
            z_random = tf.random.normal((self.n_generated_images, self.n_latent_dims))
            generator_inputs = tf.concat([z_random, labels], axis=-1)
            images_fake = self.model.generator(generator_inputs)
        
            for i_img in range(self.n_generated_images):
                ax[i_img, i_class].imshow(images_fake[i_img], cmap=self.cmap)              
                ax[i_img, i_class].axis('off')
    
            if self.class_names is not None:
                ax[0, i_class].set_title(self.class_names[i_class])
            else:
                ax[0, i_class].set_title(f'Class {i_class}')           
        
        # Save to PNG file
        fakes_save_path = os.path.join(self.output_dir, f'epoch{epoch+1:03d}_fakes.png')
        fig.savefig(fakes_save_path, facecolor='white')
        
        # Close the figure so it doesn't render on screen
        plt.close(fig)
            
class SaveImagesConditional(Callback):
    def __init__(self, output_dir, model, example_images, example_labels,
                 n_generated_images=10,
                 n_latent_dims=8,
                 class_names=None,
                 ):
        """Callback for saving examples of reconstructions and synthetic 
        images after each epoch.

        Args:
            output_dir (str): Path to save location.
            model: Model object.
            example_images (array): Numpy array containing real images for 
                which to compute reconstructions.
            example_labels (array): Numpy array containing one-hot labels.
            n_generated_images (int, optional): Number of synthetic images 
                to generate. Defaults to 10.
            n_latent_dims (int, optional): Size of latent representation. Defaults to 8.
            class_names (list, optional): Name of each class to use in figures.
        """        
        self.output_dir = output_dir
        self.example_images = example_images
        self.example_labels = example_labels
        self.n_classes = example_labels.shape[1]
        self.n_generated_images = n_generated_images
        self.n_latent_dims = n_latent_dims
        self.model = model
        self.class_names = class_names
        
        self.images_tensor = tf.convert_to_tensor(self.example_images, dtype=tf.float32)
        self.labels_tensor = tf.convert_to_tensor(self.example_labels, dtype=tf.float32)
        
    def on_epoch_end(self, epoch, logs=None):
        """Overrides the on_epoch_end method of the superclass Callback. Here,
        we define what operations should be done, as the name implies, at the
        end of each epoch. This includes saving example reconstructions and
        generating some de novo images.
        
        When overriding a superclass method, it's best to match the signature 
        (i.e. the set of arguments taken) in your new method. In this case, 
        Keras will be calling this method automatically during model training, 
        and it assumes that the method takes these exact two arguments (epoch 
        and logs).

        Args:
            epoch (int): Current epoch number. 
            logs (dict, optional): A dict of metrics from the current epoch. 
                We don't need it here, but since the original Callback.on_epoch_end 
                includes this argument, we have to include it in our overriding 
                method. Defaults to None.
        """        
        del logs # Unused
        
        # Compress images into probabilistic latent representations with
        # encoder. Note that calling the encoder model on a numpy array yields
        # outputs as tensor objects.
        encoder_inputs = self.model.make_conditional_input(self.images_tensor, self.labels_tensor)  
        z_mean, z_logvar = self.model.encoder(encoder_inputs)
        # Sample points from these probabilistic latents
        z_real = sfn(z_mean, z_logvar)
        # Reconstruct images with decoder
        decoder_inputs = tf.concat([z_real, self.labels_tensor], axis=1)
        recons = self.model.decoder(decoder_inputs)
        # Convert from tensors to numpy arrays
        recons = recons.numpy()
        n_recons = recons.shape[0]
        
        # Use the gray colormap if the image is grayscale (last dimension is 1),
        # but not if the image is RGB color (last dimension is 3).
        if self.example_images.shape[-1] == 1:
            cmap = 'gray'
        else:
            cmap = None
        
        # Create a figure showing the real image in the top row and the 
        # reconstruction in the bottom row
        fig, ax = plt.subplots(2, n_recons, figsize=(n_recons, 2))
        for i in range(n_recons):
            ax[0, i].imshow(self.example_images[i], cmap=cmap)
            ax[1, i].imshow(recons[i], cmap=cmap) 
            ax[0, i].axis('off')
            ax[1, i].axis('off')
        # Save to PNG file named with the current epoch number
        # We add 1 to the epoch so that the first epoch is 1 instead of 0  
        recon_save_path = os.path.join(self.output_dir, f'epoch{epoch+1:03d}_recons.png')
        fig.savefig(recon_save_path, facecolor='white')
                
        # Generate some synthetic images from random latent representations for
        # each class
        fig2, ax2 = plt.subplots(self.n_generated_images, self.n_classes, 
                                 figsize=(self.n_classes, self.n_generated_images))
        
        for i_class in range(self.n_classes):
            labels = np.zeros((self.n_generated_images, self.n_classes))
            labels[:, i_class] = 1
            labels = tf.convert_to_tensor(labels, dtype=tf.float32)
                            
            z_random = tf.random.normal((self.n_generated_images, self.n_latent_dims))
            decoder_inputs = tf.concat([z_random, labels], axis=-1)
            images_fake = self.model.decoder(decoder_inputs)
        
            for i_img in range(self.n_generated_images):
                ax2[i_img, i_class].imshow(images_fake[i_img], cmap=cmap)              
                ax2[i_img, i_class].axis('off')
    
            if self.class_names is not None:
                ax2[0, i_class].set_title(self.class_names[i_class])
            else:
                ax2[0, i_class].set_title(f'Class {i_class}')           
        
        # Save to PNG file
        fakes_save_path = os.path.join(self.output_dir, f'epoch{epoch+1:03d}_fakes.png')
        fig2.savefig(fakes_save_path, facecolor='white')
        
        # Close the figures so they don't render on screen
        plt.close(fig)
        plt.close(fig2)