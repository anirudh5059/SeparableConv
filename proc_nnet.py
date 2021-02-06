import tensorflow as tf
from sepconv import SpaceDepthSepConv2
from model_designs import spacedepthsepconv
import numpy as np

#Default parameters
batch_size = 256
opt = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.CategoricalCrossentropy()
  
#Load the data
(x_train, y_train),(x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train = tf.keras.utils.to_categorical(y_train,10)

#(self.x_train, self.y_train),(self.x_test, self.y_test) = tf.keras.datasets.cifar100.load_data()
#self.y_test = tf.keras.utils.to_categorical(self.y_test,100)

#Normalize the data
x_train = x_train.astype(float)
x_test = x_test.astype(float)
mean = np.mean(x_train,axis=(0,1,2,3))
std = np.std(x_train,axis=(0,1,2,3))
x_train = (x_train-mean)/(std+1e-7)
x_test = (x_test-mean)/(std+1e-7)


datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False
    )
datagen.fit(x_train)
train_dataset =  datagen.flow(x_train, y_train, batch_size=batch_size)


#Model
model = spacedepthsepconv()

#Training Parameters
epochs = 150
linf_norm = False
l2_norm=False
k = 15
data_aug = False
google_reg=False


#Training
for epoch in range(epochs):
  print("\nStart of epoch %d" % (epoch,))

# Iterate over the batches of the dataset.
  for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):


# Open a GradientTape to record the operations run
# during the forward pass, which enables auto-differentiation.
    with tf.GradientTape() as tape:

# Run the forward pass of the layer.
# The operations that the layer applies
# to its inputs are going to be recorded
# on the GradientTape.
      logits = model(x_batch_train, training=True)  # Logits for this minibatch

# Compute the loss value for this minibatch.
      loss_value = loss_fn(y_batch_train, logits)

# Use the gradient tape to automatically retrieve
# the gradients of the trainable variables with respect to the loss.
    grads = tape.gradient(loss_value, model.trainable_weights)
# Run one step of gradient descent by updating
# the value of the variables to minimize the loss.
    opt.apply_gradients(zip(grads, model.trainable_weights))

#Regularization and logging
    if(step > len(x_train)/batch_size):
      break
    if step % k == 0:
      for layer in model.layers:
        if(isinstance(layer, SpaceDepthSepConv2) and (layer.norm_flag == True)):
#print("Got here")
#layer.debug()
          if(linf_norm == True):
            layer.normalize_linf()
          if(l2_norm == True):
            layer.normalize_l2()
          if(google_reg == True):
            layer.normalize_google()
    print("Training loss (for one batch) at step %d: %.4f"% (step, float(loss_value)))

test_loss, test_acc = model.evaluate(x=self.x_test, y=self.y_test)
train_loss, train_accuracy = model.evaluate(x=self.x_train, y=self.y_train)
# Print out the model accuracy 
print('\nTrain accuracy:', train_accuracy)
print('\nTest accuracy:', test_acc)

