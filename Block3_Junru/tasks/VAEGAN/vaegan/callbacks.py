import os
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.keras.callbacks import Callback

# local def. AAM
def sfn(dist_mean, dist_logvar):
    z = tf.random.normal(shape=tf.shape(dist_mean))
    sampledZ = dist_mean + tf.exp(0.5 * dist_logvar) * z
    return sampledZ

class GenerateImages(Callback):
    def __init__(self, output_dir, model, 
                 cmap='gray',
                 n_generated_images=10, 
                 n_latent_dims=8):
        """Callback for saving examples of synthetic 
        images from a GAN object after each epoch.

        Args:
            output_dir (str): Path to save location. 
            model: Model object.
            cmap (str): colormap. Defaults to 'gray'.
            n_generated_images (int, optional): Number of synthetic images 
                to generate. Defaults to 10.
            n_latent_dims (int, optional): Size of latent representation.
                Defaults to 8.
        """        
        self.output_dir = output_dir
        self.cmap = cmap
        self.n_generated_images = n_generated_images
        self.n_latent_dims = n_latent_dims
        self.model = model
        
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
                                        
        # Generate some synthetic images from random latent representations
        z_random = tf.random.normal((self.n_generated_images, self.n_latent_dims))
        images_fake = self.model.generator(z_random)
        
        # Create a figure with these generated images
        fig, ax = plt.subplots(1, self.n_generated_images, 
                               figsize=(self.n_generated_images, 1))
        for i in range(self.n_generated_images):
            ax[i].imshow(images_fake[i], cmap=self.cmap)              
            ax[i].axis('off')
        
        # Save to PNG file
        fakes_save_path = os.path.join(self.output_dir, f'epoch{epoch+1:03d}_fakes.png')
        fig.savefig(fakes_save_path, facecolor='white')
        
        # Close the figure so it doesn't render on screen
        plt.close(fig)
  
        
class SaveImages(Callback):
    def __init__(self, output_dir, model, example_images, 
                 n_generated_images=10, 
                 n_latent_dims=8):
        """Callback for saving examples of reconstructions and synthetic 
        images after each epoch.

        Args:
            output_dir (str): Path to save location.
            model: Model object.
            example_images (array): Numpy array containing real images for 
                which to compute reconstructions.
            n_generated_images (int, optional): Number of synthetic images 
                to generate. Defaults to 10.
            n_latent_dims (int, optional): Size of latent representation. Defaults to 8.
        """        
        self.output_dir = output_dir
        self.example_images = example_images
        self.n_generated_images = n_generated_images
        self.n_latent_dims = n_latent_dims
        self.model = model
        
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
        z_mean, z_logvar = self.model.encoder(self.example_images)
        # Sample points from these probabilistic latents
        z_real = sfn(z_mean, z_logvar)
        # Reconstruct images with decoder
        recons = self.model.decoder(z_real)
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
                
        # Generate some synthetic images from random latent representations
        z_random = tf.random.normal((self.n_generated_images, self.n_latent_dims))
        images_fake = self.model.decoder(z_random)
        
        # Create a figure with these generated images
        fig2, ax2 = plt.subplots(1, self.n_generated_images, 
                                 figsize=(self.n_generated_images, 1))
        for i in range(self.n_generated_images):
            ax2[i].imshow(images_fake[i], cmap=cmap)              
            ax2[i].axis('off')
        
        # Save to PNG file
        fakes_save_path = os.path.join(self.output_dir, f'epoch{epoch+1:03d}_fakes.png')
        fig2.savefig(fakes_save_path, facecolor='white')
        
        # Close the figures so they don't render on screen
        plt.close(fig)
        plt.close(fig2)
        
class SaveModel(Callback):
    
    def __init__(self, save_dir, model, save_epochs=10):
        """Saves the model after every specified number of epochs.

        Args:
            save_dir (str): Directory where model checkpoints will be saved.
            model: Model object
            save_epochs (int, optional): How often to save the model. Defaults to 10.
        """
        
        self.save_dir = save_dir
        self.model = model
        self.save_epochs = save_epochs
        
    def on_epoch_end(self, epoch, logs=None):
        """Overrides the on_epoch_end method of the superclass Callback. Here,
        we define what operations should be done, as the name implies, at the
        end of each epoch. In thise case, we save the model in its current 
        state after every specified number of epochs. 
        
        Args:
            epoch (int): Current epoch number. 
            logs (dict, optional): A dict of metrics from the current epoch. 
                We don't need it here, but since the original Callback.on_epoch_end 
                includes this argument, we have to include it in our overriding 
                method. Defaults to None.
        """   
        
        del logs # Unused
        
        epoch += 1 # Since Keras begins epochs at 0, we add 1
        if (epoch % self.save_epochs) == 0:            
            save_path = os.path.join(self.save_dir, f'epoch{epoch:03d}_checkpoint')
            self.model.save(save_path)
    