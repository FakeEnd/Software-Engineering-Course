import tensorflow as tf
import tensorflow.keras.layers as tkl
import numpy as np


def sample_from_normal(dist_mean, dist_logvar):
    """Sample a point from a normal distribution, parameterized by a given mean 
    and log-variance.

    Args:
        dist_mean (tensor): Distribution means.
        dist_logvar (tensor): Distribution log-variances.

    Returns:
        tensor: Sampled points
    """    

    # ===== ToImplement  Exercise3a ====
    # ========================        
    # The encoder in a VAE map an imput image, x, not to a single latent vector z, i.e. a point estimate, but rather to a distribution
    # of likely z's. Typically the form of that disribution is assumed to follow a normal distrubtion parameterized by its mean and variance. 
    #
    # The second half of the VAE is the decoder. 
    # The decoder must reconstruct the original image from the latent representation, however, to do that it needs to 
    # draw a sample from the learned distribution over latent space. That is it needs to pick one those likley z's and expand it via
    # transposed convolutions.
    # 
    # To help us build the VAE, we need you to implement the sampling of a point from a normal distribution parameterized by a given me and and 
    # log-variance (learned by the encoder). These are provided as function arguments: dist_mean, and dist_logvar
    # In fact the encoder has learned a tensor of means (one for each element of z) and a tensor of log-variances (again one for each element of z)  
    # 
    # HINT: See the lecture notes and use tf.random.normal  and tf.exp
    # Store the result in sampledZ so it can be returned.
    # 

    # ========================   
    return sampledZ

def kl_divergence(dist_mean, dist_logvar):
    """Compute the closed-form KL Divergence between a given distribution and a 
    normal prior distribution with mean 0 and variance 1.

    Args:
        dist_mean (tensor): Distribution means.
        dist_logvar (tensor): Distribution log-variances.

    """    
    # ===== ToImplement Exercise3b ====
    # ========================    

    # An exact solution to the VAE is intractable therefore we turn to an approximate method (variational inferencing). 
    # This imposes a prior on our solution to regularize it.
    # In particular we impose a prior in the form of a standard normal distribution with mean 0 and variance 1.
    # Hint see the lecture notes. tf.square, tf.reduce_sum, tf.reduce_mean, tf.exp may be helful

    # 1. First compute the KL divergence within each sample. 

    # 2. Next, Sum the KL divergence within each sample
    
    # 3. Finally compute the mean KL divergence over all samples, and returen the result in divKL_AllSamples
    # e.g.  divKL_AllSamples=.... 


    # ========================    

    return divKL_AllSamples

class Encoder(tf.keras.Model):
    
    def __init__(self,
                 n_latent_dims=8,
                 name='encoder',
                 **kwargs):
        """Convolutional encoder which compresses an image into a probabilistic 
        latent representation.

        Args:
            n_latent_dims (int, optional): Length of the latent representation. 
                Defaults to 8.
            name (str, optional): Model name. Defaults to 'encoder'.
        """        
        # Call to tf.keras.Model's __init__ method. Does some model setup like
        # setting the name.
        super(Encoder, self).__init__(name=name, **kwargs)
        
        self.n_latent_dims = n_latent_dims
        
        # Define layers. Let d x d be the dimensions of the input image. 

        # Each convolution layer uses 4 x 4 kernels, and the number of filters
        # per layer comes from the filters_per_layer argument.
        self.conv0 = tkl.Conv2D(16, 4, padding='same', name='conv0')
        self.bn0 = tkl.BatchNormalization(name='bn0')
        self.relu0 = tkl.ReLU(name='relu0')
        # Since the padding of the convolutional layer was 'same' and there was
        # no striding or pooling, the output of this block should still be d x d.
        
        self.conv1 = tkl.Conv2D(32, 4, padding='same', strides=(2, 2), name='conv1')
        self.bn1 = tkl.BatchNormalization(name='bn1')
        self.relu1 = tkl.ReLU(name='relu1')
        # Applying convolution with a stride of 2 results in an output size of
        # (d/2) x (d/2).
        
        self.conv2 = tkl.Conv2D(64, 4, padding='same', strides=(2, 2), name='conv2')
        self.bn2 = tkl.BatchNormalization(name='bn2')
        self.relu2 = tkl.ReLU(name='relu2')
        # (d/4) x (d/4)
        
        self.conv3 = tkl.Conv2D(64, 4, padding='same', strides=(2, 2), name='conv3')
        self.bn3 = tkl.BatchNormalization(name='bn3')
        self.relu3 = tkl.ReLU(name='relu3')
        # (d/8) x (d/8)
        
        # Flatten 2D output into a vector, then apply dense layer
        self.flatten = tkl.Flatten(name='flatten')
        self.dense = tkl.Dense(128, name='dense')
        
        # Dense layers to output the variational mean and log-variance
        self.dense_mean = tkl.Dense(self.n_latent_dims, name='dense_mean')
        # Initializing the weights of the log-variance layer at zero seems to
        # improve numerical stability and prevent the KL divergence from blowing
        # up. 
        self.dense_logvar = tkl.Dense(self.n_latent_dims, kernel_initializer='zeros', 
                                      name='dense_logvar')       
        
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
        
        '''
        The call method is where we define the forward pass through the model.
        This is used under-the-hood during training (model.fit), testing
        (model.evaluate), and prediction (model.predict). 
        
        The first argument, inputs, is in tensor form. When the user calls fit,
        evaluate, or predict, Tensorflow will automatically convert the input
        data into a tensor if needed.
        
        The second argument, training, determines whether the model is in the
        training state or testing state. This is important because
        BatchNormalization behaves differently during training vs. testing.
        During training, it learns a running mean and s.d. of the data and uses
        this to normalize the data, then updates this mean and s.d. as it sees
        more training data. During testing, it uses the learned mean and s.d.
        without updating it.
        '''
        

        # here we implement the encoder's forward pass using the layers defined in the __init__()
        # Recall that the encoder reduces the input image to a lower dimensional latent represention
        # In this case it that representation is in the form of a mean and log variance of the distribution 
        # of likely z's (likely latent representations of the input image)
        #
        # You are given a minibatch of input images stored in the variable: inputs. So that is where you should start.
        # Recall that our pattern is convolution then batchnormalize adn then activation function (eg. relu).
        #
        # In a VAE, the encoder mirrors the decoder, ie they both have the same number of layers and weights/layer.

        # First block: images --> convolution --> batch normalize  --> relu  --> x
        # 1. Use the self.conv0() layer to extract image features from the images in the variable, inputs. Store the output in x.
        # 2. Use the self.bn0() layer to normalize the next mini batch of training images x and the training labels in the variable training. Store the output in x.
        # 3. Then transform x by applying the relu0() layer. Have it take in x from the previous step. Store the output again in x. This way we update x  and learn a dense NN layer 
        x = self.conv0(inputs)
        x = self.bn0(x, training=training)
        x = self.relu0(x)
        
        # Second block: x --> conv1 --> bn1 --> relu1  --> x
        # 1. Use the self.conv1() layer to extract image features from x that are predictive of authenticity (real vs fake) and class label. Store the output in x
        # 2. Use the self.bn1() layer to normalize the next mini batch of training images x and the training labels in the variable training. Store the output in x.
        # 3. Use the relu1 layer to transform x by applying the RELU activation to x.  Store the output again in x.
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        
        # Third block: x --> conv2 --> bn2 --> relu2  --> x
        # Otherwise similar to previous block 
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu2(x)

        # Forth block: x --> conv2 --> bn2 --> relu2  --> x
        # Otherwise similar to previous block 
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.relu3(x)
        
        # Fifth block: x --> flatten --> dense --> x
        # 1. Use the flatten() layer to flatten the tensor of image features in x to a vector.  Store the output in x
        # 2. use the dense() layer to apply a dense NN layer that learns the combinations of elements in x that are predictive of the distribution of z's that represent our input image.  Store the output in x      
        x = self.flatten(x)
        x = self.dense(x)
        
        # Sixth block: 
        # 1. use dense_mean() layer to apply a dense NN layer that learns the combinations of elements in x that are predictive of the mean of the z distrubion for our input image. store the result in z_mean.
        # 2. use dense_logvar() layer to apply a dense NN layer that learns the combinations of elements in x that are predictive of the log variance for our input image. store the result in z_logvar.
        z_mean = self.dense_mean(x)
        z_logvar = self.dense_logvar(x)


        return z_mean, z_logvar
    
    def get_config(self):
        # To allow saving and loading of a custom model, we need to implement a
        # get_config method. This should return a dict containing all important
        # arguments needed to instantiate the model.
        return {'n_latent_dims': self.n_latent_dims}
    
    @classmethod
    def from_config(cls, config):
        # Create model using config, which is a dict of arguments. Note that cls
        # refers to the current class, and calling cls is equivalent to
        # instantiating the class.
        return cls(**config)
    

class Decoder(tf.keras.Model):
    
    def __init__(self,
                 image_shape=(32, 32, 1),
                 name='decoder',
                 **kwargs):
        """Convolutional decoder which decompresses a vector latent 
        representation back into an image.

        Args:
            image_shape (tuple, optional): Image shape. Defaults to (32, 32, 1).
            name (str, optional): Model name. Defaults to 'decoder'.
        """        
        # Call to tf.keras.Model's __init__ method. Does some model setup like
        # setting the name.
        super(Decoder, self).__init__(name=name, **kwargs)
        
        self.image_shape = image_shape

        '''
        First, we need to figure out how many neurons the initial Dense layer
        needs. In order to produce the correct final image shape, the output
        shapes of the intermediate layers in this decoder needs to perfectly 
        mirror those of the encoder. 
        
        The first few layers of the decoder look like this: 
        latent vector -> dense -> reshape 1D vector to 3D tensor -> conv0
        
        The shape of the input to conv0 needs to match the shape of the output
        of the last convolution layer in the encoder (conv3). And since the 
        number of elements needs to stay the same before and after reshaping, 
        the output of the dense layer and the input to conv0 need to have the 
        same number of elements. We could compute this number manually and 
        hardcode it in, but it's better to determine it programmatically in 
        case the image shape changes.
        
        The encoder starts with an image with shape d x d, then applies 3
        convolutions with 2 x 2 striding (plus 1 non-strided convolution). That
        means the output of the last convolution (conv3) will have shape 
        
        (d/2^3) x (d/2^3) x (n_filters)
        
        Consequently, the first Dense layer needs to output a vector with this 
        number of elements. 
        
        '''
        
        n_encoder_last_filters = 64 # Number of filters in the encoder's last convolutional layer
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
        self.tconv0 = tkl.Conv2DTranspose(64, 4, strides=(2, 2), padding='same', name='tconv0')
        self.bn0 = tkl.BatchNormalization(name='bn0')
        self.relu0 = tkl.ReLU(name='relu0')
        # (d/4) x (d/4)
        
        self.tconv1 = tkl.Conv2DTranspose(64, 4, strides=(2, 2), padding='same', name='tconv1')
        self.bn1 = tkl.BatchNormalization(name='bn1')
        self.relu1 = tkl.ReLU(name='relu1')
        # (d/2) x (d/2)
        
        self.tconv2 = tkl.Conv2DTranspose(32, 4, strides=(2, 2), padding='same', name='tconv2')
        self.bn2 = tkl.BatchNormalization(name='bn2')
        self.relu2 = tkl.ReLU(name='relu2')
        # d x d
        
        self.tconv3 = tkl.Conv2DTranspose(16, 4, padding='same', name='tconv3')
        self.bn3 = tkl.BatchNormalization(name='bn3')
        self.relu3 = tkl.ReLU(name='relu3')
        
        # A final convolution layer with a sigmoid activation to produce the
        # output image. The number of filters should equal the number of color
        # channels in the image.
        n_channels = self.image_shape[-1]
        self.conv_out = tkl.Conv2D(n_channels, 1, padding='same', name='conv_out')
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
    
        x = self.conv_out(x)
        x = self.sigmoid_out(x)
        
        return x    
    
    def get_config(self):
        # To allow saving and loading of a custom model, we need to implement a
        # get_config method. This should return a dict containing all important
        # arguments needed to instantiate the model.
        return {'image_shape': self.image_shape}
    
    @classmethod
    def from_config(cls, config):
        # Create model using config, which is a dict of arguments. Note that cls
        # refers to the current class, and calling cls is equivalent to
        # instantiating the class.
        return cls(**config)
    
    
class VAE(tf.keras.Model):
    '''Variational autoencoder'''
    
    def __init__(self, 
                 n_latent_dims=8, 
                 image_shape=(32, 32, 1),
                 kl_loss_weight=1.,
                 name='vae', 
                 **kwargs):
        """Variational autoencoder-generative adversarial network. Contains a
        VAE which learns to compress and decompress an image and a
        discriminator which helps the VAE produce more realistic images.

        Args:
            n_latent_dims (int, optional): Size of latent representation.
                Defaults to 8. 
            image_shape (tuple, optional): Image shape. Defaults
                to (32, 32, 1). 
            kl_loss_weight (float, optional): Weight for KL divergence 
                regularization in the encoder. Defaults to 1.. 
            adv_loss_weight (float, optional): Weight for adversarial loss in 
                the decoder. Defaults to 1.. 
            name (str, optional): Model name. Defaults to 'vaegan'.
        """        
        

        # ===== ToImplement   Exercise3c ====
        # ========================           

        # Now that we have created new datatypes (classes) for the Encoder and Decoder we can use them in our new VAE class.
        #  want to make our VAE model conditional. That is we want to be abl to tell it which class label to generate an image of. 
        
        # 
        # 1. Call the base class init, passing it all of the relevant fields that it expects as arguments. 
        #    Hint its generally wise to pass through any misc arguments sent to the derived clas __init__() in kwargs  to the base class __init__().


        # 2. Store the number of latent dimensions, image shape and kl loss weight in a member variable of the same name in the current (self) instance variable.




        # 3. VAE Has-a Encoder  
        #    VAE has-a Decoder 
        #    Therefore construct an Encoder instance and store it in self.encoder
        #              construct a Decoder instance and store it in self.decoder
        # 


        # 4. Use the mean squared error as the reconstruction loss. Do not reduce
        # values (average over samples), return per-pixel values instead.   
        # Give it the name 'recon_mse'    
        # Therefore construct a tf.keras.losses.MeanSquaredError  and store the instance in self.loss_recon
        
        # 5. Define some custom metrics to track the running means of each loss.
        # The values of these metrics will be printed in the progress bar with
        # each training iteration.
        # Therefore construct the following metrics
        #  tf.keras.metrics.Mean  with the name 'recon_loss' and store it in self.loss_recon_tracker
        #  tf.keras.metrics.Mean  with the name 'kl_loss' and store it in self.loss_kl_tracker
        #  tf.keras.metrics.Mean  with the name 'total_loss' and store it in self.loss_total_tracker

 


        print(f"Loaded version: {__name__}")

    def call(self, inputs, training=None):

        # ===== ToImplement    Exercise3d  ====
        # ========================        
        # To give you some assistance we partailly did this one for you. 
        # HOWEVER, note that below there are no program comments. What a lazy SWE!
        # Since we need those to make our code maintainable, 
        # study what this is doing and add comments explaining what each line is doing.
        z_mean, z_logvar = self.encoder(inputs, training=training)
        z = sample_from_normal(z_mean, z_logvar)
        recons = self.decoder(z, training=training)
        return recons
    
    @property
    def metrics(self):
        '''
        Return a list of losses
        '''
        return [self.loss_recon_tracker,
                self.loss_kl_tracker,
                self.loss_total_tracker]
        
    def train_step(self, data):
        """Defines a single training iteration, including the forward pass,
        computation of losses, backpropagation, and weight updates.

        Args:
            data (tensor): Input images.

        Returns:
            dict: Loss values.
        """        


        # ===== ToImplement   Exercise3e ====
        # ========================         
        images_real = data  # Unpack any inputs. In this case we only have the predictors ie. the images since this is unsupervised
        
        # 1. Tell tensorflow to build a computational graph for the forward pass, so that gradients can be computed for us.
        #    Therefore use tf.GradientTape(persistent=True) to start the beginning of a python runtime context using the "with" keyword.
        #    Note the argument persistent=True is required to compute multiple gradients from a single GradientTape

        #      # === Beginning OF INDENTED BLOCK ===
        #      Note: steps 2- 7 are indented and thus inside the context block 
        #      2. Use encoder to predict probabilistic latent representations by invoking its forward pass, call, but do that indirectly through self.encoder
        #         provide images_real and tell it that we are training  (training=True)
        #         unpack the outputs into z_mean and z_logvar

        #      3.Sample a point from the latent distributions.  Use your sample_from_normal() with the arguments of the mean and logvar returned from self.encoder
        #        store the result in z

        #      4. Now Use decoder to reconstruct image by invoking its forward pass, call, but do taht indirectly through self.decoder
        #         provide the sampled point as input and provide the argument training=True since this is a train_step
        #         store the result in the  variable recons

        #      5. Compute KL divergence loss between latent representations and the prior using your function kl_divergence() 
        #         store the result in kl_loss

        #    [ 6. ] Compute reconstruction loss
        #        # To simplify we provide this code for you  (you need to uncomment it, but keep it indented):
        #        recon_loss_pixel = self.loss_recon(images_real, recons)   # uncomment this line

        #        # Recon loss is computed per pixel. Sum over the pixels and then
        #        # average across samples.
        #        recon_loss_sample = tf.reduce_sum(recon_loss_pixel, axis=(1, 2))  # uncomment this line
        #        recon_loss = tf.reduce_mean(recon_loss_sample)  # uncomment this line

        # #  [ 7. ] Sum the loss terms (we provide this code for you):
        #        total_loss = recon_loss + self.kl_loss_weight * kl_loss  # uncomment this line
        #        # === END OF INDENTED BLOCK ===

        # 8. Compute the gradients for each loss wrt their respectively model weights
        #    A) Use the gradient tape to compute the gradients for the encoder and store the results in grads_enc

        #    B) Use the gradient tape to compute the gradients for the dencoder and store the results in grads_dec


        # 9. Apply the gradient descent steps to each submodel. The optimizer
        # attribute is created when model.compile(optimizer) is called by the
        # user.
        # A) First apply the gradient to the encoder. use self.optimizer.apply_gradients()

        # B) Then apply the gradient to the decoder. use self.optimizer.apply_gradients()
        
        
        # [ 10. ] Update the running means of the losses
        #  To simplify we provide this code for you 
        self.loss_recon_tracker.update_state(recon_loss)
        self.loss_kl_tracker.update_state(kl_loss)
        self.loss_total_tracker.update_state(total_loss)
        
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
                'kl_loss_weight': self.kl_loss_weight}
    
    @classmethod
    def from_config(cls, config):
        # Create model using config, which is a dict of arguments. Note that cls
        # refers to the current class, and calling cls is equivalent to
        # instantiating the class.
        return cls(**config)
 

# ToImplement:  Note if the exercise3 wont run because exercise 5a,b (ConditionalVAE Below isnt complete yet, then simply comment out ConditionalVAE while doing exercise3)
class ConditionalVAE(VAE):
    def __init__(self, 
                n_latent_dims=8, 
                n_classes=10,
                image_shape=(32, 32, 1),
                kl_loss_weight=1.,
                name='cvae', 
                **kwargs):
    

        
        # Next we want to make our VAE model conditional. That is we want to be abl to tell it which class label to generate an image of. 
        # To do that we can modify our original VAE implementation.
        # While we could cut and paste (copy) the code, that would introduce code bloat. 
        # A standard practice in OOP-based deep learning is to reduce code bloat via inheritance. 
        # 
        # Above you can see that our ConditionalVAE derives (inherits) from VAE
        # 
        # Now your task is to code up the missing parts of this ConditionalVAE.


        # For this method, the constructor __init__() we will need to accept the number of classes we want to learn, n_classes.

        # 1. Store the number of classes in a member variable of the same name in the current (self) instance variable.
        self.n_classes = n_classes
        
        # 2. Now call the base class init, passing it all of the relevant fields that it expects as arguments. 
        #    Hint its generally wise to pass through any arguments sent to the derived clas __init__()  to the base class __init__().
        super().__init__(n_latent_dims=n_latent_dims,
                         image_shape=image_shape,
                         kl_loss_weight=kl_loss_weight,
                         name=name,
                         **kwargs)

        print(f"Loaded version: {__name__}")
 

    def make_conditional_input(self, images, labels):
        """Convert one-hot label vectors into a 3D tensor and concatenate to the image.
        This way every pixel in the iamge has a corresponding one-hot labeled vector.

        Args:
            self: this object instance
            images: tensor of images (the current minibatch)
            labels: one-hot label vectors for the images in the current minibatch
        """    
        # ===== ToImplement  Exercise5a ====
        # ========================        

        # Convert one-hot label vectors into a 3D tensor and concatenate to the image
        # this one is a bit tricky so we will develop it together.        

        # 1. first we need to get the number of samples, height and width of the images 
        
        # 2. next build a tensor containing the image labels for the current minibatch, which is stored in the variable, labels.


        
        # 3. Finally since the tensors are the same dimension along the last axis, we can concatenate them together. 

        # ========================


        return imagesConcatenatedWithPerPixelLabelVectors
    
    def call(self, inputs, training=None):
        """ Forward pass for the conditional VAE 
        Args:
            self: this object instance
            inputs: tuple of images and class labels for the the current minibatch
            training: training, determines whether the model is in the
                training state or testing state. This is important because
                BatchNormalization behaves differently during training vs. testing.
                During training, it learns a running mean and s.d. of the data and uses
                this to normalize the data, then updates this mean and s.d. as it sees
                more training data. During testing, it uses the learned mean and s.d.
                without updating it.
        """  
        images, classes = inputs
        

        # The forward pass is typically where a models design is most evident because 
        # it is where the  sequence of transformations in order from first to last is most succinctly implemented.

        # For the ConditionalVAE the encoder compresses both the input images and classes 
        # and the decoder expands both the latent compressed representation z and the given class label.
        #
        # Implementationwise the Forward pass:
        # 1. Uses make_conditionial_input()  gets the concatenated encoder inputs 
        # 2. uses the encoder and decoder modules in the appropriate order, via the self.encoder() and self.decoder() functions. 
        #     Python routes these function invocations to the encoder's call() method on each class object, 
        #     Thus self.encoder() runs the forward pass of the encoder and 
        #     self.decoder() runs the forward pass of the decoder.  
        # 3. Calls sample_from_normal()
        # 4. Uses tf.concat so decoder has both latent rep and class labels as its combined input.
        #
        # Hint#2: refer to VAE.call() as a rough guide for the ordering of the steps that we need here.
        # importantly, the decoder now needs an additional input, namely the class labels which we have extracted for you on line above "images, classes = inputs".
        encoder_inputs = self.make_conditional_input(images, classes)
        z_mean, z_logvar = self.encoder(encoder_inputs, training=training)
        z = sample_from_normal(z_mean, z_logvar)
        
        decoder_inputs = tf.concat([z, classes], axis=-1)        
        recons = self.decoder(decoder_inputs, training=training)


        return recons   # finally we return the reconstructed image, which is a primary output of a VAE.
    
    def train_step(self, data):
        """Defines a single training iteration, including the forward pass,
        computation of losses, backpropagation, and weight updates.

        Args:
            data (tuple of (tensor, tensor)): input images, class labels

        Returns:
            dict: Loss values.
        """        
        # Unpack the data ... for the ConditionalVAE we have both actual (real) images and their labels 
        images_real, class_real = data[0]
        
        # ===== ToImplement  Exercise5b ====        
        # 1a. Expand the class labels into 3D tensors and concatenate to the channel 
        # dimension of the images by calling your make_conditional_input. 
        # Store the result in variable encoder_inputs

        
        # 1b. Tell tensorflow to build a computational graph for the forward pass, so that gradients can be computed for us.
        #    Therefore use tf.GradientTape(persistent=True) to start the beginning of a python runtime context using the "with" keyword.
        #    Note the argument persistent=True is required to compute multiple gradients from a single GradientTape        
        #    persistent=True is required to compute multiple gradients from a single GradientTape
            
            # Follow steps 2..7 from the VAE's train_step() however there are two parts *** that are different for the ConditionalVAE:            
            # Use encoder to predict probabilistic latent representations 
            
            # Sample a point from the latent distributions. 
            
            # *** Step 3b: (new step) You will need to concatenate labels to latent representations

            # *** Step 4: Use decoder to reconstruct image from the concatenated result

            # Compute KL divergence loss between latent representations and the prior
            
            # Compute reconstruction loss

            # Recon loss is computed per pixel. Sum over the pixels and then
            # average across samples.
            
            # Sum the loss terms

                       
        # Follow steps 8-10 from the VAE.train_step()
        # 8. Compute the gradients for each loss wrt their respectively model weights
        
        # 9. Apply the gradient descent steps to each submodel. The optimizer
        # attribute is created when model.compile(optimizer) is called by the
        # user.
        
        # 10. Update the running means of the losses
        
        
        # [Given] Get the current values of these running means as a dict. These values
        # will be printed in the progress bar.
        dictLosses = {loss.name: loss.result() for loss in self.metrics}
        return dictLosses
    