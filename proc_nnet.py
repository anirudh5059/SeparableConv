import tensorflow as tf
from sepconv import SpaceDepthSepConv2
from model_designs import spacedepthsepconv
from model_designs import convnet1
from model_designs import Resnet32
from sepconv import SVD_Conv_Tensor
from sepconv import Clip_OperatorNorm
import numpy as np

def normalize_google_Conv2D(kern, in_shape):
  L =  tf.math.reduce_max(SVD_Conv_Tensor(kern, in_shape))
  return tf.math.divide(kern, L)

#Default parameters
batch_size = 256
#lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#    initial_learning_rate=1e-3,
#    decay_steps=3000,
#    decay_rate=0.96)
#lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
#    boundaries = [3000, 6000, 12000], values = [0.001, 0.0005, 0.0002, 0.00001])
opt = tf.keras.optimizers.Adam()
#opt = tf.keras.optimizers.SGD(learning_rate = lr_schedule, momentum = 0.9)
loss_fn = tf.keras.losses.CategoricalCrossentropy()
  
#Load the data
(x_train, y_train),(x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train = tf.keras.utils.to_categorical(y_train,10)
y_test = tf.keras.utils.to_categorical(y_test,10)

#(self.x_train, self.y_train),(self.x_test, self.y_test) = tf.keras.datasets.cifar100.load_data()

#Normalize the data
x_train = x_train.astype(float)
x_test = x_test.astype(float)
mean = np.mean(x_train,axis=(0,1,2,3))
std = np.std(x_train,axis=(0,1,2,3))
x_train = (x_train-mean)/(std+1e-7)
x_test = (x_test-mean)/(std+1e-7)

#Data augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False
    )
datagen.fit(x_train)
train_dataset =  datagen.flow(x_train, y_train, batch_size=batch_size)



#Training Parameters
epochs = 50
linf_norm = False
l2_norm=False
k = 50
google_reg=True


  # Training
for bound in [1,5,10,20,30,50]:
  # Model
  model = spacedepthsepconv()
  model.compile(optimizer='adam', loss='categorical_crossentropy', run_eagerly=True, metrics=['accuracy'])
  max_ = 0
  o_path = "../experiments/SSDC"+str(bound)+".txt"
  f = open(o_path, "x")
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


      # Break manually because the data generator loops indefinitely
      if(step > len(x_train)/batch_size):
        break

      # Regularization and logging
      if (step % k == 0):
        for layer in model.layers:
          if(isinstance(layer, SpaceDepthSepConv2) and (layer.norm_flag == True)):
            if(linf_norm == True):
              layer.normalize_linf()
            if(l2_norm == True):
              layer.normalize_l2()
            if(google_reg == True):
              layer.normalize_google(bound)
        print("Training loss (for one batch) at step %d: %.4f"% (step, float(loss_value)))

    # Last epoch, check operator norm of every convolutional layer
    if (epoch == epochs-1):
      for layer in model.layers:
        if(isinstance(layer, tf.keras.layers.Conv2D)):
          arr = layer.get_weights()
          in_shape = layer.input_shape[1:3]
          D, U, V, conv_shape = SVD_Conv_Tensor(arr[0], in_shape)
          norm = tf.math.reduce_max(D)
          if max_ < norm:
            max_ = norm
          # print("Layer "+layer.name+": nnnormalized norm = "+str(norm))
          # if(google_reg == True):
          #   n_kern = Clip_OperatorNorm(D, U, V, conv_shape, bound)
          #   layer.set_weights([n_kern, arr[1]])
      print("Maximum operator norm seen post the last epoch is: " + str(max_))
  test_loss, test_accuracy = model.evaluate(x=x_test, y=y_test)
  train_loss, train_accuracy = model.evaluate(x=x_train, y=y_train)
  # Print out the model accuracy 
  print('\nTrain accuracy:', train_accuracy)
  print('\nTest accuracy:', test_accuracy)
  f.write("Test Accuracy: "+str(test_accuracy))
  f.write("Train Accuracy: "+str(train_accuracy))
  f.write("Max norm: "+str(max_))
  f.close()

