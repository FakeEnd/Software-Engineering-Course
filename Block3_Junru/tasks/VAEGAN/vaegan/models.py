import tensorflow as tf
import tensorflow.keras.layers as tkl
import numpy as np


from .vae import sample_from_normal, kl_divergence, Encoder, Decoder
from .gan import Discriminator


# ToImplement Exercise6b
# ======================
# implement __init__, call and train_step

# VAE~GAN with 3 conv layers/module (enc & dec), pixel recon loss, non conditional
class VAEGAN(tf.keras.Model):
    
    # These are CLASS attributes, which we create here in the class namespace,
    # outside of any method. Unlike INSTANCE attributes, which are created
    # inside __init__ or other methods, the values of these attributes are
    # constant across all instances. Class attributes are often used to define
    # constants or default values. Here, we're using class attributes to tell
    # VAEGAN which classes correspond to the 3 submodels. This makes it easier
    # to switch them out later if we modify the architectures and come up with
    # different submodel classes (see VAEGANLarger below).
    encoder_class = Encoder
    decoder_class = Decoder
    discriminator_class = Discriminator
    
    def __init__(self, 
                 n_latent_dims=8, 
                 image_shape=(32, 32, 1),
                 recon_loss_weight=1.,
                 kl_loss_weight=1.,
                 adv_loss_weight=1.,
                 name='vaegan', 
                 **kwargs):
        """Variational autoencoder-generative adversarial network. Contains a
        VAE which learns to compress and decompress an image and a
        discriminator which helps the VAE produce more realistic images.

        Args:
            n_latent_dims (int, optional): Size of latent representation.
                Defaults to 8. 
            image_shape (tuple, optional): Image shape. Defaults
                to (32, 32, 1). 
            recon_loss_weight (float, optional): Weight for reconstruction 
                loss. Defaults to 1.. 
            kl_loss_weight (float, optional): Weight for KL divergence 
                regularization in the encoder. Defaults to 1.. 
            adv_loss_weight (float, optional): Weight for adversarial loss in 
                the decoder. Defaults to 1.. 
            name (str, optional): Model name. Defaults to 'vaegan'.
        """        
        


        # For you ToImlpement 

    def call(self, inputs, training=None):
        # For you ToImlpement 
        return recons
                
    @property
    def metrics(self):
        '''
        Return a list of losses
        '''
        return [self.loss_recon_tracker,
                self.loss_kl_tracker,
                self.loss_enc_tracker,
                self.loss_dec_tracker,
                self.loss_disc_tracker]
    
    def train_step(self, data):
        """Defines a single training iteration, including the forward pass,
        computation of losses, backpropagation, and weight updates.

        Args:
            data (tensor): Input images.

        Returns:
            dict: Loss values.
        """        
        images_real = data
        
        # persistent=True is required to compute multiple gradients from a single GradientTape

            # Use encoder to predict probabilistic latent representations 
            
            # Sample a point from the latent distributions. 
            
            # Use decoder to reconstruct image
            
            # Compute KL divergence loss between latent representations and the prior
            
            # Compute reconstruction loss

            # Recon loss is computed per pixel. Sum over the pixels and then
            # average across samples.
            
            # Synthesize some new images by having the decoder generate from
            # random latent vectors.
            
            # Create label vectors, 1 for real and 0 for fake/reconstruction images
            
            # Concatenate real, reconstruction, and fake images 
            
            # Predict with the discriminator
            
            # Compute discriminator classification loss
            
            # Encoder loss includes the KL divergence and the reconstruction loss

            
            # Decoder loss includes the reconstruction loss and adversarial loss
            # (negative discriminator loss since the decoder wants the
            # discriminator to predict wrongly)

                       
        # Compute the gradients for each loss wrt their respectively model weights

        
        # Apply the gradient descent steps to each submodel. The optimizer
        # attribute is created when model.compile(optimizer) is called by the
        # user.
          
        
        # Update the running means of the losses

        
        # Get the current values of these running means as a dict. These values
        # will be printed in the progress bar.
        dictLosses = {loss.name: loss.result() for loss in self.metrics}
        return dictLosses

    def get_config(self):
        # To allow saving and loading of a custom model, we need to implement a
        # get_config method. This should return a dict containing all important
        # arguments needed to instantiate the model.
        return {'n_latent_dims': self.n_latent_dims,
                'image_shape': self.image_shape,
                'recon_loss_weight': self.recon_loss_weight,
                'kl_loss_weight': self.kl_loss_weight,
                'adv_loss_weight': self.adv_loss_weight}
    
    @classmethod
    def from_config(cls, config):
        # Create model using config, which is a dict of arguments. Note that cls
        # refers to the current class, and calling cls is equivalent to
        # instantiating the class.
        return cls(**config)
    

# Try a Larger VAE~GAN with 6 conv layers rather than only 3
# =====================================================
'''
Below, let's create a "larger" VAE-GAN with more filters per layer in order to
handle the more complex CIFAR10 images. Instead of rewriting the Encoder,
Decoder, and Discriminator classes just to change a few layer parameters, we can
subclass the original classes and just override the constructor (__init__)
method where the layers are defined. 

We can then create a new VAE-GAN by subclassing the original VAEGAN class and
changing the class attributes which tell it which submodel classes to use.
'''
    
    
class EncoderLarger(Encoder):
    def __init__(self,
                 n_latent_dims=32,
                 name='encoder',
                 **kwargs):
        """A larger encoder with more convolutional filters, subclassing from
        the original Encoder. Since the layers are all defined in the
        constructor (__init__), we only need to override this method. Everything
        else can stay the same.

        Args:
            n_latent_dims (int, optional): Length of the latent representation. 
                Defaults to 32.
            name (str, optional): Model name. Defaults to 'encoder'.
        """        
        # This line calls __init__ method of the superclass of Encoder, not
        # EncoderLarger. In other words, we're calling the __init__ method of
        # the grandparent (tf.keras.Model)
        super(Encoder, self).__init__(name=name, **kwargs)
        
        self.n_latent_dims = n_latent_dims
        
        # Define layers. Let d x d be the dimensions of the input image. 
        self.conv0 = tkl.Conv2D(32, 4, padding='same', strides=(2, 2), name='conv0')
        self.bn0 = tkl.BatchNormalization(name='bn0')
        self.relu0 = tkl.ReLU(name='relu0')
        # (d/2) x (d/2).
        
        self.conv1 = tkl.Conv2D(32, 4, padding='same', name='conv1')
        self.bn1 = tkl.BatchNormalization(name='bn1')
        self.relu1 = tkl.ReLU(name='relu1')
        
        self.conv2 = tkl.Conv2D(64, 4, padding='same', strides=(2, 2), name='conv2')
        self.bn2 = tkl.BatchNormalization(name='bn2')
        self.relu2 = tkl.ReLU(name='relu2')     
        # (d/4) x (d/4)
        
        self.conv3 = tkl.Conv2D(64, 4, padding='same', name='conv3')
        self.bn3 = tkl.BatchNormalization(name='bn3')
        self.relu3 = tkl.ReLU(name='relu3')
       
        self.conv4 = tkl.Conv2D(128, 4, padding='same', strides=(2, 2), name='conv4')
        self.bn4 = tkl.BatchNormalization(name='bn4')
        self.relu4 = tkl.ReLU(name='relu4')
        # (d/8) x (d/8)
        
        self.conv5 = tkl.Conv2D(128, 4, padding='same', name='conv5')
        self.bn5 = tkl.BatchNormalization(name='bn5')
        self.relu5 = tkl.ReLU(name='relu5')
        
        # Flatten 2D output into a vector, then apply dense layer
        self.flatten = tkl.Flatten(name='flatten')
        
        # Dense layers to output the variational mean and log-variance
        self.dense_mean = tkl.Dense(self.n_latent_dims, name='dense_mean')
        self.dense_logvar = tkl.Dense(self.n_latent_dims, name='dense_logvar')
        
    def call(self, inputs, training=None):
        """Forward pass.

        Args:
            inputs (tensor): Batch of image inputs.
            training (bool, optional): Whether the model is training or 
                testing model. Defaults to None.
                
        Returns:
            (tensor, tensor): tuple containing the probabilistic latent 
                representation, which includes a set of means and a set 
                of log-variances.
        """     
        
        x = self.conv0(inputs)
        x = self.bn0(x, training=training)
        x = self.relu0(x)
        
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu2(x)
        
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.relu3(x)
        
        x = self.conv4(x)
        x = self.bn4(x, training=training)
        x = self.relu4(x)
        
        x = self.conv5(x)
        x = self.bn5(x, training=training)
        x = self.relu5(x)
        
        x = self.flatten(x)
        
        z_mean = self.dense_mean(x)
        z_logvar = self.dense_logvar(x)
        
        return z_mean, z_logvar
        
        
class DecoderLarger(Decoder):
    def __init__(self,
                 image_shape=(32, 32, 1),
                 name='decoder',
                 **kwargs):
        """A larger decoder with more convolutional filters, subclassing from
        the original Decoder. Since the layers are all defined in the
        constructor (__init__), we only need to override this method. Everything
        else can stay the same.

        Args:
            image_shape (tuple, optional): Image shape. Defaults to (32, 32, 1).
            name (str, optional): Model name. Defaults to 'decoder'.
        """        
        # This line calls __init__ method of the superclass of Decoder, not
        # DecoderLarger. In other words, we're calling the __init__ method of
        # the grandparent (tf.keras.Model)
        super(Decoder, self).__init__(name=name, **kwargs)
        
        self.image_shape = image_shape
        
        n_encoder_last_filters = 128 # Number of filters in the encoder's last convolutional layer
        n_encoder_strided_layers = 3
        first_conv_input_shape = (self.image_shape[0] // (2 ** n_encoder_strided_layers),
                                  self.image_shape[1] // (2 ** n_encoder_strided_layers),
                                  n_encoder_last_filters)
        
        dense_neurons = np.product(first_conv_input_shape)
        
        self.dense = tkl.Dense(dense_neurons, name='dense')
        self.relu_dense = tkl.ReLU(name='relu_dense')
        # Reshape from 1D to 3D
        self.reshape = tkl.Reshape(first_conv_input_shape, name='reshape')
        # (d/8) x (d/8)
        
        # To do the opposite of the convolutions in the encoder, we use transposed convolutions. 
        self.tconv0 = tkl.Conv2DTranspose(128, 4, strides=(2, 2), padding='same', name='tconv0')
        self.bn0 = tkl.BatchNormalization(name='bn0')
        self.relu0 = tkl.ReLU(name='relu0')
        # (d/4) x (d/4)
        
        self.tconv1 = tkl.Conv2DTranspose(128, 4, padding='same', name='tconv1')
        self.bn1 = tkl.BatchNormalization(name='bn1')
        self.relu1 = tkl.ReLU(name='relu1')
                
        self.tconv2 = tkl.Conv2DTranspose(64, 4, strides=(2, 2), padding='same', name='tconv2')
        self.bn2 = tkl.BatchNormalization(name='bn2')
        self.relu2 = tkl.ReLU(name='relu2')
        # (d/2) x (d/2)
                
        self.tconv3 = tkl.Conv2DTranspose(64, 4, padding='same', name='tconv3')
        self.bn3 = tkl.BatchNormalization(name='bn3')
        self.relu3 = tkl.ReLU(name='relu3')
                
        self.tconv4 = tkl.Conv2DTranspose(32, 4, strides=(2, 2), padding='same', name='tconv4')
        self.bn4 = tkl.BatchNormalization(name='bn4')
        self.relu4 = tkl.ReLU(name='relu4')
        # d x d
                
        self.tconv5 = tkl.Conv2DTranspose(32, 4, padding='same', name='tconv5')
        self.bn5 = tkl.BatchNormalization(name='bn5')
        self.relu5 = tkl.ReLU(name='relu5')
        
        # A final convolution layer with a sigmoid activation to produce the
        # output image. The number of filters should equal the number of color
        # channels in the image.
        n_channels = self.image_shape[-1]
        self.conv_out = tkl.Conv2D(n_channels, 3, padding='same', name='conv_out')
        self.sigmoid_out = tkl.Activation('sigmoid', name='sigmoid_out')
        
    def call(self, inputs, training=None):
        """Forward pass.

        Args:
            inputs (tensor): Batch of inputs containing the latent means 
                and log-variances.
            training (bool, optional): Whether the model is training or 
                testing model. Defaults to None.
                
        Returns:
            tensor: reconstructed image
        """                             
        
        x = self.dense(inputs)
        x = self.relu_dense(x)
        x = self.reshape(x)
        
        x = self.tconv0(x)
        x = self.bn0(x, training=training)
        x = self.relu0(x)
        
        x = self.tconv1(x)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        
        x = self.tconv2(x)
        x = self.bn2(x, training=training)
        x = self.relu2(x)
        
        x = self.tconv3(x)
        x = self.bn3(x, training=training)
        x = self.relu3(x)
    
        x = self.tconv4(x)
        x = self.bn4(x, training=training)
        x = self.relu4(x)
    
        x = self.tconv5(x)
        x = self.bn5(x, training=training)
        x = self.relu5(x)
    
        x = self.conv_out(x)
        x = self.sigmoid_out(x)
        
        return x  


class DiscriminatorLarger(Discriminator):
    def __init__(self, 
                 name='discriminator', 
                 **kwargs):
        """A larger discrminator with more convolutional filters, subclassing from
        the original Discriminator. Since the layers are all defined in the
        constructor (__init__), we only need to override this method. Everything
        else can stay the same.

        Args:
            name (str, optional): Model name. Defaults to 'discriminator'.
        """
        # This line calls __init__ method of the superclass of Discriminator,
        # not DiscriminatorLarger. In other words, we're calling the __init__
        # method of the grandparent (tf.keras.Model)        
        super(Discriminator, self).__init__(name=name, **kwargs)
        
        # Define the layers
        self.conv0 = tkl.Conv2D(128, 4, padding='same', name='conv0')
        self.bn0 = tkl.BatchNormalization(name='bn0')
        self.relu0 = tkl.ReLU(name='relu0')
        
        self.conv1 = tkl.Conv2D(128, 4, strides=(2, 2), padding='same', name='conv1')
        self.bn1 = tkl.BatchNormalization(name='bn1')
        self.relu1 = tkl.ReLU(name='relu1')
        
        self.conv2 = tkl.Conv2D(256, 4, strides=(2, 2), padding='same', name='conv2')
        self.bn2 = tkl.BatchNormalization(name='bn2')
        self.relu2 = tkl.ReLU(name='relu2')
        
        self.flatten = tkl.Flatten(name='flatten')
        self.dense = tkl.Dense(512, name='dense')
        self.relu_dense = tkl.ReLU(name='relu_dense')
        
        self.dense_out = tkl.Dense(1, name='dense_out')
        self.sigmoid_out = tkl.Activation('sigmoid', name='sigmoid_out')
        
# VAE~GAN with a 6 conv layers/module (enc & dec), pixel recon loss, non conditional
class VAEGANLarger(VAEGAN):
    
    # Creating a new VAEGAN subclass that uses the new encoder, decoder, and
    # discriminator subclasses is easy. We just need to override these class
    # attributes. The rest of the VAEGAN class should work the same as before.
    
    encoder_class = EncoderLarger
    decoder_class = DecoderLarger
    discriminator_class = DiscriminatorLarger  
    

# ToImplement Exercise6b
# ======================
# OPTIONAL.  You may choose to instead implement __init__, call and train_step  of the VAEGANSimMetric


# VAE~GAN with a 6 conv layers/module (enc & dec), pixel recon and semantic feature loss, non conditional    
class VAEGANSimMetric(VAEGANLarger):
    '''
    Uses discriminator feature similarity as an additional reconstruction loss.
    During the forward pass, the features produced by the last convolutional
    layer of the discriminator are obtained for the real and reconstructed
    images. The new feature-based reconstruction loss is the MSE betwen these
    feature representations.
    
    This is based on the "learned similarity metric" for VAE-GANs proposed by
    Larsen et al. 2016 https://arxiv.org/pdf/1512.09300.pdf and the perceptual
    similarity metrics proposed by Dosovitskiy and Brox 2016
    https://arxiv.org/pdf/1602.02644.pdf.
    '''
    
    def __init__(self, 
                 n_latent_dims=8, 
                 image_shape=(32, 32, 1),
                 pixel_loss_weight=1.,
                 feature_loss_weight=1.,
                 kl_loss_weight=1.,
                 adv_loss_weight=1.,
                 name='vaegan', 
                 **kwargs):
        
        # For you ToImlpement
        
        print("in VAEGANSimMetric __init__7")
     
    @property    
    def metrics(self):
        '''
        Return a list of losses and metrics.
        '''
        superclass_metrics = super().metrics
        # Concatenate the additional feature similarity loss
        return superclass_metrics + [self.loss_feature_tracker]
    
    def train_step(self, data):
        """Defines a single training iteration, including the forward pass,
        computation of losses, backpropagation, and weight updates.

        Args:
            data (tensor): Input images.

        Returns:
            dict: Loss values.
        """        
        images_real = data
        
        # persistent=True is required to compute multiple gradients from a single GradientTape

            # Use encoder to predict probabilistic latent representations 
            
            # Sample a point from the latent distributions. 
            
            # Use decoder to reconstruct image
            
            # Compute KL divergence loss between latent representations and the prior
            
            # Compute reconstruction loss

            # Recon loss is computed per pixel. Sum over the pixels and then
            # average across samples.
            
            # Synthesize some new images by having the decoder generate from
            # random latent vectors.
            
            # Create label vectors, 1 for real and 0 for fake/reconstruction images
            
            # Concatenate real, reconstruction, and fake images 
            
            # Predict with the discriminator

            
            # Compute discriminator classification loss

            
            # Get the discriminator feature maps for the real and recon images

            
            # Compute MSE between these feature maps

                        
            # Encoder loss includes the KL divergence, pixel-based
            # reconstruction loss, and feature similarity loss

            
            # Decoder loss includes the pixel-based reconstruction loss, feature
            # similarity loss, and adversarial loss (negative discriminator loss
            # since the decoder wants the discriminator to predict wrongly)

                       
        # Compute the gradients for each loss wrt their respectively model weights

        
        # Apply the gradient descent steps to each submodel. The optimizer
        # attribute is created when model.compile(optimizer) is called by the
        # user.
          
        
        # Update the running means of the losses

        
        # Get the current values of these running means as a dict. These values
        # will be printed in the progress bar.
        dictLosses = {loss.name: loss.result() for loss in self.metrics}
        return dictLosses
