"""We now build the CNN model in real MNIST data set.
We should have a good testing performance since the data is no longer random
"""
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow.keras.datasets.mnist import load_data
"""
# Set limit to TesorFlow so it only use half of a GPU card
gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_virtual_device_configuration(
    gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7500)]
)
"""
# normalize data to zero mean and unit variance
def normalize_data(x):
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    return (x - x_mean) / x_std 

class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation="relu")
        self.flatten = Flatten()
        self.d1 = Dense(128, activation="relu")
        self.d2 = Dense(10)

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

def test_step(images, labels):
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

if __name__ == "__main__":
    # load the MNIST data
    (x_train, y_train), (x_test, y_test) = load_data()

    # Normalize the data
    x_train = normalize_data(x_train)
    x_test = normalize_data(x_test)

    # Add a channels dimension
    x_train = x_train[..., tf.newaxis].astype("float32")
    x_test = x_test[..., tf.newaxis].astype("float32")

    # Make training/testing dataset
    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
    )

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

    # Create an instance of the model, define loss function and Adam optimizier
    model = MyModel()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")
    test_loss = tf.keras.metrics.Mean(name="test_loss")
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")

    EPOCHS = 5
    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for images, labels in train_ds:
            train_step(images, labels)

        for test_images, test_labels in test_ds:
            test_step(images, test_labels)

        print(
            f"Epoch {epoch + 1}, "
            f"Train Loss: {train_loss.result()}, "
            f"Train Accuracy: {train_accuracy.result() * 100}, "
            f"Test Loss: {test_loss.result()}, "
            f"Test Accuracy: {test_accuracy.result() * 100}"
        )
    if test_accuracy.result() * 100 < 97:
        print("Test failed, your model have very low accuracy, please check it again")
    else:
        print("Test passed, your model have very good accuracy")

#There are 2 bugs (might be more) in this script
#Bug3, hint: place a conditional break breakpoint test_labels.shape[0] == 16 at line 94
#Bug4, hint: WATCH x_train, find out when it starts to have incorrect values