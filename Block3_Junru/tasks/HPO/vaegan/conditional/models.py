import tensorflow as tf
import tensorflow.keras.layers as tkl
from ..gan_SOLUTIONS import fParseActivationFunctions
from ..models_SOLUTIONS import Discriminator, VAEGAN, sample_from_normal, kl_divergence

class ConditionalVAEGAN(VAEGAN):
    
    def __init__(self, 
                 n_classes=10,              
                 image_shape=(32, 32, 1),    
                 n_latent_dims=4,           
                 recon_loss_weight=1.,      
                 kl_loss_weight=1.,         
                 adv_loss_weight=1.,        
                 name='cvaegan', 
                 encoder_params = {},       
                 decoder_params = {},
                 discriminator_params = {},
                 **kwargs):
        
        self.n_classes = n_classes
        
        super().__init__(n_latent_dims=n_latent_dims,
                         image_shape=image_shape,
                         recon_loss_weight=recon_loss_weight,
                         kl_loss_weight=kl_loss_weight,
                         adv_loss_weight=adv_loss_weight,
                         encoder_params = encoder_params,
                         decoder_params = decoder_params,
                         discriminator_params = discriminator_params,
                         name=name,
                         **kwargs)
        
    def make_conditional_input(self, images, labels):
        # Convert one-hot label vectors into a 3D tensor and concatenate to the image
        
        n_samples = tf.shape(images)[0]
        height = tf.shape(images)[1]
        width = tf.shape(images)[2]
        
        labels_tensor = tf.reshape(labels, [-1, 1, 1, self.n_classes])
        ones_tensor = tf.ones([n_samples, height, width, self.n_classes])
        labels_tensor = labels_tensor * ones_tensor
        
        return tf.concat([images, labels_tensor], axis=-1)
    
    def call(self, inputs, training=None):
        images, classes = inputs
        encoder_inputs = self.make_conditional_input(images, classes)
        
        z_mean, z_logvar = self.encoder(encoder_inputs, training=training)
        z = sample_from_normal(z_mean, z_logvar)
        
        decoder_inputs = tf.concat([z, classes], axis=-1)        
        recons = self.decoder(decoder_inputs, training=training)
        return recons
    
    def train_step(self, data):
        """Defines a single training iteration, including the forward pass,
        computation of losses, backpropagation, and weight updates.

        Args:
            data (tuple of (tensor, tensor)): input images, class labels

        Returns:
            dict: Loss values.
        """        
        images_real, class_real = data[0]
        
        encoder_inputs = self.make_conditional_input(images_real, class_real)
        
        # persistent=True is required to compute multiple gradients from a single GradientTape
        with tf.GradientTape(persistent=True) as gt:
            # Use encoder to predict probabilistic latent representations 
            z_mean, z_logvar = self.encoder(encoder_inputs, training=True)
            
            # Sample a point from the latent distributions. 
            z = sample_from_normal(z_mean, z_logvar)
            
            # Concatenate labels to latent representations
            z_conditional = tf.concat([z, class_real], axis=1)
            
            # Use decoder to reconstruct image
            recons = self.decoder(z_conditional, training=True)
            
            # Compute KL divergence loss between latent representations and the prior
            kl_loss = kl_divergence(z_mean, z_logvar)
            
            # Compute reconstruction loss
            recon_loss_pixel = self.loss_recon(images_real, recons)
            # Recon loss is computed per pixel. Sum over the pixels and then
            # average across samples.
            recon_loss_sample = tf.reduce_sum(recon_loss_pixel, axis=(1, 2))
            recon_loss = tf.reduce_mean(recon_loss_sample)
            
            # Synthesize some new images by having the decoder generate from
            # random latent vectors with random labels.
            n_samples = tf.shape(images_real)[0]
            # Random latent vectors from normal distribution
            z_random = tf.random.normal((n_samples, self.n_latent_dims))            
            # Random class labels from uniform distribution
            class_probs = tf.ones((1, self.n_classes))
            class_random = tf.random.categorical(class_probs, num_samples=n_samples)
            class_random = tf.squeeze(class_random)
            # Convert from categorical to one-hot
            class_random = tf.one_hot(class_random, depth=self.n_classes, dtype=tf.float32)

            z_random_conditional = tf.concat([z_random, class_random], axis=1)
            
            images_fake = self.decoder(z_random_conditional, training=True)
            
            # Create label vectors, 1 for real and 0 for fake/reconstruction images
            labels_real = tf.ones((n_samples, 1))
            labels_fake = tf.zeros((n_samples, 1))
            
            # Concatenate real, reconstruction, and fake images 
            images_concat = tf.concat([images_real, recons, images_fake], axis=0)
            labels_concat = tf.concat([labels_real, labels_fake, labels_fake], axis=0)
            
            # Predict with the discriminator
            labels_pred = self.discriminator(images_concat, training=True)
            
            # Compute discriminator classification loss
            disc_loss = self.loss_disc(labels_concat, labels_pred)
            
            # Encoder loss includes the KL divergence and the reconstruction loss
            encoder_loss = kl_loss * self.kl_loss_weight + recon_loss * self.recon_loss_weight
            
            # Decoder loss includes the reconstruction loss and adversarial loss
            # (negative discriminator loss since the decoder wants the
            # discriminator to predict wrongly)
            decoder_loss = recon_loss * self.recon_loss_weight - disc_loss * self.adv_loss_weight
                       
        # Compute the gradients for each loss wrt their respectively model weights
        grads_enc = gt.gradient(encoder_loss, self.encoder.trainable_weights)
        grads_dec = gt.gradient(decoder_loss, self.decoder.trainable_weights)
        grads_disc = gt.gradient(disc_loss, self.discriminator.trainable_weights)
        
        # Apply the gradient descent steps to each submodel. The optimizer
        # attribute is created when model.compile(optimizer) is called by the
        # user.
        self.optimizer.apply_gradients(zip(grads_enc, self.encoder.trainable_weights))
        self.optimizer.apply_gradients(zip(grads_dec, self.decoder.trainable_weights))
        self.optimizer.apply_gradients(zip(grads_disc, self.discriminator.trainable_weights))           
        
        # Update the running means of the losses
        self.loss_recon_tracker.update_state(recon_loss)
        self.loss_kl_tracker.update_state(kl_loss)
        self.loss_enc_tracker.update_state(encoder_loss)
        self.loss_dec_tracker.update_state(decoder_loss)
        self.loss_disc_tracker.update_state(disc_loss)
        
        # Get the current values of these running means as a dict. These values
        # will be printed in the progress bar.
        dictLosses = {loss.name: loss.result() for loss in self.metrics}
        return dictLosses
    

class MultiTaskDiscriminator(Discriminator):
    
    def __init__(self, 
                 n_classes,
                 discriminator_conv_layer_units = [32,64,64],
                 discriminator_conv_layer_kernel_size = [4,4,4],
                 discriminator_conv_layer_strides = [(1,1), (2,2), (2,2)],
                 discriminator_conv_layer_activation_functions = ['relu','relu', 'relu'],
                 discriminator_dense_layer_units = [128],
                 generator_dense_layer_units = [],
                 name='discriminator', 
                 **kwargs):
        """Discriminator for classifying real from fake/reconstructed images.

        Args:
            name (str, optional): Model name. Defaults to 'discriminator'.
        """        
        super(Discriminator, self).__init__(name=name, **kwargs)
        
        self.n_classes = n_classes
        
        # Define the layers for the Discriminator module that we will later link together in the call() method.
                # Parse activation function lists for 
        self.discriminator_conv_layer_activation_functions = fParseActivationFunctions(discriminator_conv_layer_activation_functions)
        self.discriminator_layers = []

         # Conv layer blocks
        for i in range( len(discriminator_conv_layer_units) ):
            self.discriminator_layers += [tkl.Conv2D(discriminator_conv_layer_units[i], discriminator_conv_layer_kernel_size[i], padding = 'same', 
                                       strides = discriminator_conv_layer_strides[i], name = f"conv{i}")]
            self.discriminator_layers += [tkl.BatchNormalization(name=f'bn{i}')]
            self.discriminator_layers += [tkl.Activation(self.discriminator_conv_layer_activation_functions[i][1],
                                                         name=f'{self.discriminator_conv_layer_activation_functions[i][0]}{i}')]
            
        # Flatten 2D output into a vector, then apply dense layer(s)
        self.discriminator_layers += [tkl.Flatten(name='flatten')]
        
        # Dense Layer blocks
        for i in range(len(discriminator_dense_layer_units)):
            self.discriminator_layers += [tkl.Dense(discriminator_dense_layer_units[i],name = f"dense{i}")]

        self.dense_real = tkl.Dense(1, name='dense_real')
        self.sigmoid_real = tkl.Activation('sigmoid', name='sigmoid_real')
        
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
        x = inputs
        for layer in self.discriminator_layers:
            x = layer(x, training = training)
        
        y_real = self.dense_real(x)
        y_real = self.sigmoid_real(y_real)
    
        y_class = self.dense_class(x)
        y_class = self.softmax_class(y_class)
        
        if return_features:
            return y_real, y_class, x
        else:
            return y_real, y_class
    
    def get_config(self):
        return {'n_classes': self.n_classes}
    

class ConditionalVAECGAN(ConditionalVAEGAN):
    
    discriminator_class = MultiTaskDiscriminator
    
    def __init__(self, 
                 n_latent_dims=4, 
                 n_classes=10,
                 image_shape=(32, 32, 1),
                 recon_loss_weight=1.,
                 kl_loss_weight=1.,
                 adv_loss_weight=1.,
                 cond_loss_weight=1.,
                 encoder_params = {},
                 decoder_params = {},
                 discriminator_params = {},
                 name='cvaecgan', 
                 **kwargs):
            
        super(VAEGAN, self).__init__(name=name, **kwargs)
        
        self.n_latent_dims = n_latent_dims
        self.n_classes = n_classes
        self.image_shape = image_shape
        self.recon_loss_weight = recon_loss_weight
        self.kl_loss_weight = kl_loss_weight
        self.adv_loss_weight = adv_loss_weight
        self.cond_loss_weight = cond_loss_weight
        
        self.encoder = self.encoder_class(n_latent_dims=self.n_latent_dims, **encoder_params)
        self.decoder = self.decoder_class(image_shape=self.image_shape, **decoder_params)
        self.discriminator = self.discriminator_class(n_classes, **discriminator_params)

        # Use the mean squared error as the reconstruction loss. Do not reduce
        # values (average over samples), return per-pixel values instead.       
        self.loss_recon = tf.keras.losses.MeanSquaredError(name='recon_mse', reduction='none')
        
        # Use binary cross-entropy for the discrminator's classification loss
        self.loss_disc = tf.keras.losses.BinaryCrossentropy(name='disc_bce')
        
        # Define some custom metrics to track the running means of each loss.
        # The values of these metrics will be printed in the progress bar with
        # each training iteration.
        self.loss_recon_tracker = tf.keras.metrics.Mean(name='recon_loss')
        self.loss_kl_tracker = tf.keras.metrics.Mean(name='kl_loss')
        self.loss_enc_tracker = tf.keras.metrics.Mean(name='enc_loss')
        self.loss_dec_tracker = tf.keras.metrics.Mean(name='dec_loss')
        self.loss_disc_tracker = tf.keras.metrics.Mean(name='disc_loss')
                
        # Categorical cross-entropy loss for discriminator classification
        self.loss_class = tf.keras.losses.CategoricalCrossentropy(name='class_cce')
        self.loss_class_tracker = tf.keras.metrics.Mean(name='class_loss')
        
        # Classification accuracy 
        self.metric_class = tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='top_1_acc')
    
    @property
    def metrics(self):
        '''
        Return a list of losses
        '''
        return [self.loss_recon_tracker,
                self.loss_kl_tracker,
                self.loss_class_tracker,
                self.loss_enc_tracker,
                self.loss_dec_tracker,
                self.loss_disc_tracker,
                self.metric_class]
    
    def train_step(self, data):
        """Defines a single training iteration, including the forward pass,
        computation of losses, backpropagation, and weight updates.

        Args:
            data (tuple of (tensor, tensor)): input images, class labels

        Returns:
            dict: Loss values.
        """        
        images_real, class_real = data[0]
        
        encoder_inputs = self.make_conditional_input(images_real, class_real)
        
        # persistent=True is required to compute multiple gradients from a single GradientTape
        with tf.GradientTape(persistent=True) as gt:
            # Use encoder to predict probabilistic latent representations 
            z_mean, z_logvar = self.encoder(encoder_inputs, training=True)
            
            # Sample a point from the latent distributions. 
            z = sample_from_normal(z_mean, z_logvar)
            
            # Concatenate labels to latent representations
            z_conditional = tf.concat([z, class_real], axis=1)
            
            # Use decoder to reconstruct image
            recons = self.decoder(z_conditional, training=True)
            
            # Compute KL divergence loss between latent representations and the prior
            kl_loss = kl_divergence(z_mean, z_logvar)
            
            # Compute reconstruction loss
            recon_loss_pixel = self.loss_recon(images_real, recons)
            # Recon loss is computed per pixel. Sum over the pixels and then
            # average across samples.
            recon_loss_sample = tf.reduce_sum(recon_loss_pixel, axis=(1, 2))
            recon_loss = tf.reduce_mean(recon_loss_sample)
            
            # Synthesize some new images by having the decoder generate from
            # random latent vectors with random labels.
            n_samples = tf.shape(images_real)[0]
            # Random latent vectors from normal distribution
            z_random = tf.random.normal((n_samples, self.n_latent_dims))            
            # Random class labels from uniform distribution
            class_probs = tf.ones((1, self.n_classes))
            class_random = tf.random.categorical(class_probs, num_samples=n_samples)
            class_random = tf.squeeze(class_random)
            # Convert from categorical to one-hot
            class_random = tf.one_hot(class_random, depth=self.n_classes, dtype=tf.float32)

            z_random_conditional = tf.concat([z_random, class_random], axis=1)
            
            images_fake = self.decoder(z_random_conditional, training=True)
            
            # Create label vectors, 1 for real and 0 for fake/reconstruction images
            labels_real = tf.ones((n_samples, 1))
            labels_fake = tf.zeros((n_samples, 1))
            
            # Concatenate real, reconstruction, and fake images 
            images_concat = tf.concat([images_real, recons, images_fake], axis=0)
            labels_concat = tf.concat([labels_real, labels_fake, labels_fake], axis=0)
            class_concat = tf.concat([class_real, class_real, class_random], axis=0)
            
            # Predict with the discriminator
            labels_pred, class_pred = self.discriminator(images_concat, training=True)
            
            # Compute discriminator authenticity prediction (real/fake) loss
            auth_loss = self.loss_disc(labels_concat, labels_pred)
            
            # Compute discriminator classification loss for real images only
            class_pred_reals = class_pred[:n_samples, :]
            class_loss_real = self.loss_class(class_real, class_pred_reals)
            # Accuracy
            self.metric_class.update_state(class_real, class_pred_reals)
            
            # Compute discriminator classification loss for real and fake images
            class_loss_all = self.loss_class(class_concat, class_pred)
            
            # Add the two discriminator losses together
            disc_loss = auth_loss + class_loss_real
            
            # Encoder loss includes the KL divergence and the reconstruction loss
            encoder_loss = kl_loss * self.kl_loss_weight + recon_loss * self.recon_loss_weight
            
            # Decoder loss includes the reconstruction loss, negative
            # authenticity loss, and the classification loss
            decoder_loss = recon_loss * self.recon_loss_weight \
                + class_loss_all * self.cond_loss_weight - auth_loss * self.adv_loss_weight
                       
        # Compute the gradients for each loss wrt their respectively model weights
        grads_enc = gt.gradient(encoder_loss, self.encoder.trainable_weights)
        grads_dec = gt.gradient(decoder_loss, self.decoder.trainable_weights)
        grads_disc = gt.gradient(disc_loss, self.discriminator.trainable_weights)
        
        # Apply the gradient descent steps to each submodel. The optimizer
        # attribute is created when model.compile(optimizer) is called by the
        # user.
        self.optimizer.apply_gradients(zip(grads_enc, self.encoder.trainable_weights))
        self.optimizer.apply_gradients(zip(grads_dec, self.decoder.trainable_weights))
        self.optimizer.apply_gradients(zip(grads_disc, self.discriminator.trainable_weights))           
        
        # Update the running means of the losses
        self.loss_recon_tracker.update_state(recon_loss)
        self.loss_kl_tracker.update_state(kl_loss)
        self.loss_enc_tracker.update_state(encoder_loss)
        self.loss_dec_tracker.update_state(decoder_loss)
        self.loss_disc_tracker.update_state(disc_loss)
        self.loss_class_tracker.update_state(class_loss_real)
        
        # Get the current values of these running means as a dict. These values
        # will be printed in the progress bar.
        dictLosses = {loss.name: loss.result() for loss in self.metrics}
        return dictLosses
    
 