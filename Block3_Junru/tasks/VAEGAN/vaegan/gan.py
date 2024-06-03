import tensorflow as tf
import tensorflow.keras.layers as tkl
import numpy as np


class Generator(tf.keras.Model):
    
    def __init__(self,
                 image_shape=(32, 32, 1),
                 name='generator',
                 **kwargs):
        """Convolutional generator which produces an image from a latent vector.

        Args:
            image_shape (tuple, optional): Image shape. Defaults to (32, 32, 1).
            name (str, optional): Model name. Defaults to 'generator'.
        """        
        # Call to tf.keras.Model's __init__ method. Does some model setup like
        # setting the name.
        super().__init__(name=name, **kwargs)
        
        self.image_shape = image_shape

        '''
        First, we need to figure out how many neurons the initial Dense layer
        needs in order to produce the desired output image shape.
        
        The first few layers of the generator look like this: latent vector ->
        dense -> reshape 1D vector to 3D tensor -> conv0
        
        There are 2 convolutions with 2 x 2 striding (plus 1 non-strided
        convolution). That means that the input to layer conv0 should have shape 
        
        (d/2^2) x (d/2^2) x (n_filters)
        
        Consequently, the first Dense layer needs to output a vector with this
        number of elements. 
            
        '''
        
        n_first_layer_filters = 128 # Number of filters/channels in the input to first convolutional layer
        n_strided_layers = 2
        first_conv_input_shape = (self.image_shape[0] // (2 ** n_strided_layers),
                                  self.image_shape[1] // (2 ** n_strided_layers),
                                  n_first_layer_filters)
        
        dense_neurons = np.product(first_conv_input_shape)
        
        self.dense = tkl.Dense(dense_neurons, name='dense')
        self.relu_dense = tkl.LeakyReLU(name='relu_dense')
        # Reshape from 1D to 3D
        self.reshape = tkl.Reshape(first_conv_input_shape, name='reshape')
        # (d/4) x (d/4)

        self.tconv0 = tkl.Conv2DTranspose(128, 4, strides=(2, 2), padding='same', name='conv0')
        self.relu0 = tkl.LeakyReLU(name='relu0')
        # (d/2) x (d/2)
                
        self.tconv1 = tkl.Conv2DTranspose(128, 4, strides=(2, 2), padding='same', name='conv1')
        self.relu1 = tkl.LeakyReLU(name='relu1')
        # d x d
                
        # A final convolution layer with a sigmoid activation to produce the
        # output image. The number of filters should equal the number of color
        # channels in the image.
        n_channels = self.image_shape[-1]
        self.conv_out = tkl.Conv2D(n_channels, 3, padding='same', name='conv_out')
        self.sigmoid_out = tkl.Activation('sigmoid', name='sigmoid_out')
        
    def call(self, inputs, training=None):
        """Forward pass.

        Args:
            inputs (tensor): Batch of inputs containing a compressed representation.
            training (bool, optional): Whether the model is training or 
                testing model. Defaults to None.
                
        Returns:
            tensor: reconstructed image
        """                             
        
        

        # First block: inputs --> dense  --> relu_dense --> reshape --> x
        # 1. Use the self.dense() layer to process the latent vector representation which is stored in the variable, inputs and store the result in the variable, x.
        # 2. Use transform x by applying the relu_dense() layer. Have it take in x from the previous step. Store the output again in x. This way we update x.  and learn a dense NN layer 
        #     This learns a linear combination of the elements of the latent vector, inputs, and then applies a nonlinear relu activation function afterwards so that we learn interesting combinations. 
        # 3. Use reshape() layer to make x into a 2D image sutable for the next block. Store the result in x. 
        x = self.dense(inputs)
        x = self.relu_dense(x)
        x = self.reshape(x)
        
        # Second block: x --> tconv0 --> relu0 --> x
        # 1. Use the tconv0() layer to perform a trasnsposed convolution on x so that we learn an expansion that constructs a slightly larger image with more detail. Store the result in x. 
        # 2. Use the relu0() layer to apply a nonlinear relu activation to the transposed result from the previous step.
        x = self.tconv0(x)
        x = self.relu0(x)

        # Third block: x --> tconv1 --> relu1 --> x
        # 1. Use the tconv1() layer to perform a trasnsposed convolution on x so that we learn an expansion that constructs a slightly larger image with more detail. Store the result in x. 
        # 2. Use the relu1() layer to apply a nonlinear relu activation to the transposed result from the previous step.
        x = self.tconv1(x)
        x = self.relu1(x)

        # Fourth block: x --> conv_out --> sigmoid  --> x
        # 1. Use the conv_out() layer to perform a convolution on x. Store the result in x. 
        # 2. Use the sigmoid_out() layer to apply a sigmoid relu activation to the result from the previous step. Sigmoid values range from 0..1  which are readily mapped to the image intensities we aim to reconstruct.
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
    

class Discriminator(tf.keras.Model):
    def __init__(self, 
                 name='discriminator', 
                 **kwargs):
        """Discriminator for classifying real from fake images.

        Args:
            name (str, optional): Model name. Defaults to 'discriminator'.
        """        
        super(Discriminator, self).__init__(name=name, **kwargs)
        
        # Define the layers
        self.conv0 = tkl.Conv2D(32, 4, padding='same', name='conv0')
        self.bn0 = tkl.BatchNormalization(name='bn0')
        self.relu0 = tkl.ReLU(name='relu0')
        
        self.conv1 = tkl.Conv2D(64, 4, strides=(2, 2), padding='same', name='conv1')
        self.bn1 = tkl.BatchNormalization(name='bn1')
        self.relu1 = tkl.ReLU(name='relu1')
        
        self.flatten = tkl.Flatten(name='flatten')
        
        self.dense_out = tkl.Dense(1, name='dense_out')
        self.sigmoid_out = tkl.Activation('sigmoid', name='sigmoid_out')
        
    def call(self, inputs, training=None):
        """Define the forward pass, which produces a predicted classification.

        Args:
            inputs (tensor): Image input.
            training (bool, optional): Whether the model is training or 
                testing model. Defaults to None.
                
        Returns:
            tensor: predicted classification
        """        
        x = self.conv0(inputs)
        x = self.bn0(x, training=training)
        x = self.relu0(x)
        
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        
        x = self.flatten(x)
        
        y = self.dense_out(x)
        y = self.sigmoid_out(y)
        
        return y
            
    def get_config(self):
        # To allow saving and loading of a custom model, we need to implement a
        # get_config method. This should return a dict containing all important
        # arguments needed to instantiate the model.
        return {}
    
    @classmethod
    def from_config(cls, config):
        # Create model using config, which is a dict of arguments. Note that cls
        # refers to the current class, and calling cls is equivalent to
        # instantiating the class.
        return cls(**config)


class GAN(tf.keras.Model):
    '''Generative adversarial network'''


    def __init__(self, 
                 n_latent_dims=128, 
                 image_shape=(32, 32, 1),
                 generator_lr=0.0001,
                 discriminator_lr=0.00001,
                 name='gan', 
                 **kwargs):
        """Generative adversarial network containing a generator to synthesize 
        images and an adversary to to discriminate between real and fake images.

        Args:
            n_latent_dims (int, optional): Size of latent representation.
                Defaults to 8. 
            image_shape (tuple, optional): Image shape. Defaults
                to (32, 32, 1). 
            generator_lr (float, optional): Adam learning rate for generator. 
                Defaults to 0.0001.
            discriminator_lr (float, optional): Adam learning rate for discriminator. 
                Defaults to 0.00001.
            name (str, optional): Model name. Defaults to 'gan'.
        """        
        
        # ===== ToImplement   Exercise5c ====
        # ===================================  
        # 
        # 1. Call base class constructor      

        
        # 2. store the 4 necessary attributes provided to the constructor. 
        #    Use as your attribute names, the same names as in the constructor input arguments




        # 3. Use the has-a mechanism to contain two construct necessary instances and store them in self.generator and self.discriminator       



        # 4. Use binary cross-entropy for the discrminator's classification loss. Store this in self.loss_bce
        # Look up the keras function losses.BinaryCrossentropy. Give it a suitable name.


        # 5. Create a custom metric objects to track the running means of each loss.
        # The values will be printed in the progress bar with each training
        # iteration.
        # store them in self.loss_gen_tracker and self.loss_disc_tracker.
        # Find a suitable metric from tf.keras.metrics



        # 6. Create Adam optimizers to do the gradient descent 
        #    Store them in self.optimizer_gen and self.optimizer_disc



        print(f"Loaded version: {__name__}")

    # ===== ToImplement   Exercise5d ====
    # ===================================
    # Implement the GAN forward pass method
    #  0. Write the def ... statement and then Write an appropriate doc string
    #  1. there should be 3 arguments to this method. 
    #     one of them is inputs  which contains the data to do the forward pass upon.
    #  2. Set n =  the number of samples by selecting the appropriate dimension using tf.shape
    #  3. Sample from a normal distribution, see tf.random documentation for a suitable function
    #     you will want n x n_latent_dims  samples from this normal distn.
    #  4. To generate fake images pass that sample through the forward pass of the generator, but also tell it whether we are in training mode or not, (from the GAN forward pass method input argument).
    #  5. Return the generated fake images.             








    @property
    def metrics(self):
        '''
        Return a list of losses
        '''
        return [self.loss_gen_tracker,
                self.loss_disc_tracker]
    
    # ===== ToImplement   Exercise5e ====
    # ===================================
    # Implement the training step method, since GANs require specialized training.
    # 0. Write the def... statement and then write an appropriate doc string 
    # 1. there are two input arguments. One of them is the tensor, images_real, with the current minibatch to train upon.
        
        # Part 1: Train the discriminator
        # 2. Generate images from random latent vectors.
        #  2a) Set n =  the number of samples by selecting the appropriate dimension using tf.shape
        #  2b) Sample from a normal distribution, see tf.random documentation for a suitable function
        #     you will want n x n_latent_dims  samples from this normal distn.  
        #  2c) Use the generator forward pass, i.e, via self.generator() to create a variable with fake images
        #     (images_fake) using the sample from the normal distn you just created in the previous substep, 2b). 
        #     Also specify that training=False 
        # 




        # 3. Create label vectors varaible, labels_real, containing ones
        #   Also create a label vector, labels_fake, containing zeros.
        #    e.g. hint for part you may want to use tf.ones



        # 4. Concatenate real and fake images into new variable,  images_disc. Hint: See tf.concat
        #    Also concatenate real and fake labels into new varaible, labels_disc.



        # 5. start a GraidentTape with default arguments


            # 6. indented: Predict with the discriminator's forward pass via self.discriminator()
            #    store the predictions in variable, labels_pred 
            #    Specify that now, training=True

            
            # 7. indented: Compute discriminator classification loss
            #    in a new variable, disc_loss, compute the binary cross entropy loss using one of the attributes of self

                                   
        # 8. NOT indented: Compute the gradient of the lost wrt the discriminator weights
        #  in a new variable, grads_disc store the gradients of disc_loss with respect to the discriminator's trainable weights  

        
        # 9. Apply the weight updates
        #    use self.optimizer_disc to apply the weight updates. Hint: use zip

        
        # Part 2: Train the generator
        # 10. start a GraidentTape with default arguments. Use a different tape context variable than used above.           


            # [ 11. indented: ] Generate images from random latent vectors. Generate twice as many
            # images as the batch size so that the generator sees as many
            # samples as the discriminator did. 
            #  To help you along, we give you this part 



            # 12. indented: Predict with the discriminator's forward pass the labels for images_fake
            #     set training=False. Store the prediction results in new variable, labels_pred

            
            # 13. indented: We want to the discriminator to think these images are real, so we
            # calculate the loss between these predictions and the "real image" labels
            # 13a)  build a new variable, labels_gen, which is a vector of ones of size 2*n_samples x 1
            # 13b)  build a new variable, gen_loss containing the binary cross entropy loss.



        # 14. NOT indented: compute the gradient of the lost wrt the generator weights
        #  in a new variable, grads_gen store the gradients of gen_loss with respect to the generator's trainable weights         

        
        # 15. Apply the weight updates
        #    use self.optimizer_gen to apply the weight updates. 

        
        # 16. Update the running means of the losses including loss_gen_tracker and loss_disc_tracker



        # [ 17. ] Get the current values of these running means as a dict. These values
        # will be printed in the progress bar.
        # To help you along this is given. Just uncomment the "##" lines below
        ##dictLosses = {loss.name: loss.result() for loss in self.metrics}

        # return the dictionary of losses
        ##return dictLosses

    def get_config(self):
        # To allow saving and loading of a custom model, we need to implement a
        # get_config method. This should return a dict containing all important
        # arguments needed to instantiate the model.
        return {'n_latent_dims': self.n_latent_dims,
                'image_shape': self.image_shape,
                'generator_lr': self.generator_lr,
                'discriminator_lr': self.discriminator_lr}
    
    @classmethod
    def from_config(cls, config):
        # Create model using config, which is a dict of arguments. Note that cls
        # refers to the current class, and calling cls is equivalent to
        # instantiating the class.
        return cls(**config)
    
class MultiTaskDiscriminator(Discriminator):
    
    def __init__(self, 
                 n_classes,
                 name='discriminator', 
                 **kwargs):
        """Discriminator for classifying real from fake/reconstructed images.  

        Args:
            name (str, optional): Model name. Defaults to 'discriminator'.
        """        
        super(Discriminator, self).__init__(name=name, **kwargs)
        
        self.n_classes = n_classes
        
        # Define the layers for the Discriminator module that we will later link together in the call() method.

        # Layers needed for first block: 
        # 1. self.conv0 ... define a 2D convolution layer via the Conv2D() function with 32 filters, each 4x4, padding same, name conv0
        # 2. self.bn0 ... define a batch normalization layer via BatchNormalization() with the name bn0
        # 3. self.relu0 ... define a RELU layer via ReLu() with the name relu0
        self.conv0 = tkl.Conv2D(32, 4, padding='same', name='conv0')
        self.bn0 = tkl.BatchNormalization(name='bn0')
        self.relu0 = tkl.ReLU(name='relu0')
        
        # Layers needed for second block: 
        # 1. self.conv1 ... define a 2D convolution layer via the Conv2D() function with 64 filters, each 4x4, 2x2 striding, padding same, name conv1
        # 2. self.bn1 ... define a batch normalization layer via BatchNormalization() with the name bn1
        # 3. self.relu1 ... define a RELU layer via ReLU() with the name relu1        
        self.conv1 = tkl.Conv2D(64, 4, strides=(2, 2), padding='same', name='conv1')
        self.bn1 = tkl.BatchNormalization(name='bn1')
        self.relu1 = tkl.ReLU(name='relu1')

        # Layers needed for third block: 
        # 1. self.flatten ... define a flattening layer via Flatten() with the name flatten
        self.flatten = tkl.Flatten(name='flatten')

        # Layers needed for block 4a: this predicts whether the image is real or fake
        # 1. self.dense_real ... define a dense layer via the Dense() function with 1 neauron and the name dense_real
        # 2. self.sigmoid_real ... define a sigmoid activation function layer via the Activation() function with the 'sigmoid' method and name sigmoid_real
        self.dense_real = tkl.Dense(1, name='dense_real')
        self.sigmoid_real = tkl.Activation('sigmoid', name='sigmoid_real')
        
        # Layers needed for block 4b: this predicts the class label of the image 
        # 1. self.dense_class ... define a dense layer via the Dense() function with n_classes and the name dense_class
        # 2. self.softmax_class  ... define a softmax activation  layer via the Softmax() function with the name softmax_class
        self.dense_class = tkl.Dense(self.n_classes, name='dense_class')
        self.softmax_class = tkl.Softmax(name='softmax_class')


    def call(self, inputs, training=None, return_features=False):
        """Define the forward pass, which produces a predicted classification.

        Args:
            inputs (tensor): Image input.
            training (bool, optional): Whether the model is training or 
                testing model. Defaults to None.
            return_features (bool, optional): Return the output of the last 
                convolutional layer. Defaults to False.
                
        Returns:
            tensor: predicted classification
        """        

   
        # First block: images & training labels --> --> convolution --> batch normalize  --> relu  --> x
        # 1. Use the self.conv0() layer to extract image features from the images in the variable, inputs. Store the output in x.
        # 2. Use the self.bn0() layer to normalize the next mini batch of training images x and the training labels in the variable training. Store the output in x.
        # 3. Then transform x by applying the relu0() layer. Have it take in x from the previous step. Store the output again in x. This way we update x  and learn a dense NN layer 
        x = self.conv0(inputs)
        x = self.bn0(x, training=training)
        x = self.relu0(x)
        
        # Second block: x --> conv1 --> bn1 --> relu1  --> x
        # 1. Use the self.conv1() layer to extract image features from x that are predictive of authenticity (real vs fake) and class label. Store the output in x
        # 2. Use the self.bn1 layer to normalzie the images in x and training labels in training. Store the output in x
        # 3. Use the relu1 layer to transform x by applying the RELU activation to x.  Store the output again in x.
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        
        # Third block: x --> flatten --> x
        # 1. Use the flatten layer to flattend the tensor of image features in x to a vector.  Store the output in x
        x = self.flatten(x)
        
        # Block 4a  x --> dense_real --> y_real;    y_real --> sigmoid_real --> y_real 
        # 1. use the dense_real layer to apply a dense NN layer that learns the combinations of elements in x that are predictive of authenticity. Store the result in y_real.
        # 2. Use the sigmoid_real layer to map the information stored in y_real to a probabilistic autheticity prediction of real (1.0) or fake (0.0)
        y_real = self.dense_real(x)
        y_real = self.sigmoid_real(y_real)
        
        # Block 4a  x --> dense_real --> y_real;    y_real --> sigmoid_real --> y_real 
        # 1. use the dense_class layer to apply a dense NN layer to learn the combinations of elements in x that are predictive of class label. Store the result in y_class.
        # 2. Use the softmax_class layer to map the information stored in y_class to a prediction of the class label (one-hot encoded). Store the result in y_class
        y_class = self.dense_class(x)
        y_class = self.softmax_class(y_class)
        


        if return_features:
            return y_real, y_class, x
        else:
            return y_real, y_class
    
    def get_config(self):
        return {'n_classes': self.n_classes}
    

class ConditionalGAN(GAN):
    '''Conditional generative adversarial network'''
    
    def __init__(self, 
                 n_classes=10,
                 cond_loss_weight=1.,
                 name='cgan', 
                 **kwargs):
        """Generative adversarial network containing a generator to synthesize 
        images and an adversary to to discriminate between real and fake images.

        Args:
            n_latent_dims (int, optional): Size of latent representation.
                Defaults to 8. 
            n_classes (int, optional): Number of classes. Defaults to 10. 
            image_shape (tuple, optional): Image shape. Defaults
                to (32, 32, 1). 
            cond_loss_weight (float, optional): Weight of conditional loss for 
                generator. Defaults to 1.
            generator_lr (float, optional): Adam learning rate for generator. 
                Defaults to 0.0001.
            discriminator_lr (float, optional): Adam learning rate for discriminator. 
                Defaults to 0.00001.
            name (str, optional): Model name. Defaults to 'gan'.
        """        
        
        # ToImplement Exercise6a_part1 ==
        # ===============================
        super().__init__(name=name, **kwargs)

        self.n_classes = n_classes
        self.cond_loss_weight = cond_loss_weight
        
        self.discriminator = MultiTaskDiscriminator(n_classes)
        
        # Categorical cross-entropy loss for discriminator classification
        #==>  complete this line of code         

        # Classification accuracy 
        #==>  complete this line of code         
     
        print(f"Loaded version: {__name__}")


    def call(self, inputs, training=None):
        # ToImplement Exercise6a_part2 ==
        # ===============================
        #==>  complete this method , roughly 5 to 7 lines of code
        
        return images_fake
        
    @property
    def metrics(self):
        '''
        Return a list of losses
        '''
        return [self.loss_gen_tracker,
                self.loss_disc_tracker,
                self.metric_class]
    
    def generate_random_classes(self, n_samples):
        # Generate random class labels from uniform distribution
        class_probs = tf.ones((1, self.n_classes))
        class_random = tf.random.categorical(class_probs, num_samples=n_samples)
        class_random = tf.squeeze(class_random)
        # Convert from categorical to one-hot
        class_random = tf.one_hot(class_random, depth=self.n_classes, dtype=tf.float32)
        return class_random
    
    ##def train_step(self, data):
        # """Defines a single training iteration, including the forward pass,
        # computation of losses, backpropagation, and weight updates.

        # Args:
        #     data (tensor): Input images.

        # Returns:
        #     dict: Loss values.
        # """        
        ##images_real, class_real = data[0]

        # ToImplement Exercise6a_part3 ==
        # ===============================        
        #==>  complete this method, you can uncomment the ## lines above and below as hints


        # Step 1: Train the discriminator

        # Generate images from random latent vectors.





        # Create label vectors, 1 for real and 0 for fake images



        # Concatenate real and fake images 



        ##with tf.GradientTape() as gt:
            # Predict with the discriminator
            
            # Compute discriminator loss for distinguishing real/fake images

            # Compute discriminator loss for predicting image class, for real images only

            # Add losses

            
            # Compute classification metric

                                   
        # Compute the gradient of the lost wrt the discriminator weights

        
        # Apply the weight updates

        
        # Step 2: Train the generator
                    
        ##with tf.GradientTape() as gt2:
            # Generate images from random latent vectors. Generate twice as many
            # images as the batch size so that the generator sees as many
            # samples as the discriminator did. 




            # Predict with the discriminator

            
            # We want to the discriminator to think these images are real, so we
            # calculate the loss between these predictions and the "real image"
            # labels


            # Compute loss between discriminator-predicted classes and the desired classes

            # Add losses

                        
        # Compute the gradient of the lost wrt the generator weights

        
        # Apply the weight updates

        
        # Update the running means of the losses
        
        # Get the current values of these running means as a dict. These values
        # will be printed in the progress bar.

        ## return dictLosses

    def get_config(self):
        # To allow saving and loading of a custom model, we need to implement a
        # get_config method. This should return a dict containing all important
        # arguments needed to instantiate the model.
        return {'n_latent_dims': self.n_latent_dims,
                'image_shape': self.image_shape,
                'generator_lr': self.generator_lr,
                'discriminator_lr': self.discriminator_lr,
                'n_classes': self.n_classes,
                'cond_loss_weight': self.cond_loss_weight}