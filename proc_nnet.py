import tensorflow as tf
from sepconv import SpaceDepthSepConv2
from model_designs import spacedepthsepconv
from model_designs import convnet1
import numpy as np

#Helper functions
def SVD_Conv_Tensor(conv, inp_shape):
  """ Find the singular values of the linear transformation
  corresponding to the convolution represented by conv on
  an n x n x depth input. """
  conv_tr = tf.cast(tf.transpose(conv, perm=[2, 3, 0, 1]), tf.complex64)
  conv_shape = conv.shape
  padding = tf.constant([[0, 0], [0, 0],
                         [0, inp_shape[0] - conv_shape[0]],
                         [0, inp_shape[1] - conv_shape[1]]])
  transform_coeff = tf.signal.fft2d(tf.pad(conv_tr, padding))
  singular_values = tf.linalg.svd(tf.transpose(transform_coeff, perm = [2, 3, 0, 1]),
                           compute_uv=False)
  return singular_values

def Clip_OperatorNorm(conv, inp_shape, clip_to):
  conv_tr = tf.cast(tf.transpose(conv, perm=[2, 3, 0, 1]), tf.complex64)
  conv_shape = conv.shape
  padding = tf.constant([[0, 0], [0, 0],
                         [0, inp_shape[0] - conv_shape[0]],
                         [0, inp_shape[1] - conv_shape[1]]])
  transform_coeff = tf.signal.fft2d(tf.pad(conv_tr, padding))
  D, U, V = tf.linalg.svd(tf.transpose(transform_coeff, perm = [2, 3, 0, 1]))
  norm = tf.reduce_max(D)
  D_clipped = tf.cast(tf.minimum(D, clip_to), tf.complex64)
  clipped_coeff = tf.linalg.matmul(U, tf.linalg.matmul(tf.linalg.diag(D_clipped),
                                         V, adjoint_b=True))
  clipped_conv_padded = tf.math.real(tf.signal.ifft2d(
      tf.transpose(clipped_coeff, perm=[2, 3, 0, 1])))
  return tf.slice(tf.transpose(clipped_conv_padded, perm=[2, 3, 0, 1]),
                  [0] * len(conv_shape), conv_shape), norm

def normalize_google_Conv2D(kern, in_shape):
  L =  tf.math.reduce_max(SVD_Conv_Tensor(kern, in_shape))
  return tf.math.divide(kern, L)

#Default parameters
batch_size = 256
opt = tf.keras.optimizers.Adam()
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
    horizontal_flip=True,
    vertical_flip=False
    )
datagen.fit(x_train)
train_dataset =  datagen.flow(x_train, y_train, batch_size=batch_size)


#Model
model = convnet1()

#Training Parameters
epochs = 40
linf_norm = False
l2_norm=False
k = 10
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
          if(linf_norm == True):
            layer.normalize_linf()
          if(l2_norm == True):
            layer.normalize_l2()
          if(google_reg == True):
            layer.normalize_google()
        if(isinstance(layer, tf.keras.layers.Conv2D)):
          arr = layer.get_weights()
          in_shape = layer.input_shape[1:3]
          n_kern, norm = Clip_OperatorNorm(arr[0], in_shape, 1)
          print("Unnormalized norm = "+str(norm))
          if(google_reg == True):
            layer.set_weights([n_kern, arr[1]])
    print("Training loss (for one batch) at step %d: %.4f"% (step, float(loss_value)))

test_loss, test_acc = model.evaluate(x=x_test, y=y_test)
train_loss, train_accuracy = model.evaluate(x=x_train, y=y_train)
# Print out the model accuracy 
print('\nTrain accuracy:', train_accuracy)
print('\nTest accuracy:', test_acc)
