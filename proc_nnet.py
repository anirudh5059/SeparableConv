import tensorflow as tf
from sepconv import SpaceDepthSepConv2
from sepconv import SpaceSepConv2
from sepconv import DepthSepConv2
from model_designs import spacedepthsepconv2
from model_designs import convnet1
from model_designs import convnet2
from model_designs import convnet2_c100
from model_designs import Resnet32
from model_designs import Resnet32_c100
from model_designs import spacesepconv1
from model_designs import depthsepconv1
from sepconv import full_svd
from sepconv import singular_values
from sepconv import Clip_OperatorNorm
from sepconv import Normalize_kern
from sepconv import l2_bound
from sepconv import linf_bound
import numpy as np
import time

#Fix memory error where cuDNN failed to initialize 
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

#Default parameters
batch_size = 256
#lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#    initial_learning_rate=1e-3,
#    decay_steps=3000,
#    decay_rate=0.96)

#Optimal for convnet2_l2
lr_c1_l2_exp = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
  boundaries = [4000, 7500], values = [0.0001, 0.00005, 0.00001])

#lr_c1_l2 = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
#  boundaries = [4000, 8000], values = [0.0001, 0.00001, 0.000005])

#Optimal for convnet2_base
lr_c1 = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
  boundaries = [4000, 12000], values = [0.001, 0.0005, 0.0001])

#Optimal for Resnet_l2
lr_res = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
  boundaries = [40000], values = [0.00001, 0.000005])

#lr_cg = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
#  boundaries = [4000, 12000], values = [0.0001, 0.00005, 0.00001])

#Optimal learning rate for Resnet_base:
#opt = tf.keras.optimizers.Adam(0.00005)

opt = tf.keras.optimizers.Adam(lr_c1_l2_exp)

#Optimal for depthsepconv1:
#opt = tf.keras.optimizers.Adam()


#opt = tf.keras.optimizers.SGD(learning_rate = lr_c1, momentum = 0.9)
loss_fn = tf.keras.losses.CategoricalCrossentropy()
  
#Load the data
#(x_train, y_train),(x_test, y_test) = tf.keras.datasets.cifar10.load_data()
#y_train = tf.keras.utils.to_categorical(y_train,10)
#y_test = tf.keras.utils.to_categorical(y_test,10)

(x_train, y_train),(x_test, y_test) = tf.keras.datasets.cifar100.load_data()
y_train = tf.keras.utils.to_categorical(y_train,100)
y_test = tf.keras.utils.to_categorical(y_test,100)

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
    zoom_range = 0.2,
    shear_range = 0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
    )

datagen.fit(x_train)
train_dataset =  datagen.flow(x_train, y_train, batch_size=batch_size)
tmp = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset_base = tmp.shuffle(buffer_size=1024).batch(batch_size)

#Training Parameters
epochs = 150
linf_norm = False
l2_norm=True
k = 2
google_reg_norm=False
google_reg_clip=False
bound_logging=False

def train_step(data_iter):

  steps_per_epoch =  len(x_train)//batch_size
  for step, (x_batch_train, y_batch_train) in enumerate(data_iter):


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
    if(step > steps_per_epoch):
      break

    # interactive print
    if(step % 50 == 0):
      print("Training loss (for one batch) at step %d: %.4f"% (step, float(loss_value)))


    # Regularization
    if (step % k == 0):
      for layer in model.layers:
        if(isinstance(layer, tf.keras.layers.Conv2D)):
          arr = layer.get_weights()
          in_shape = layer.input_shape[1:3]
          if(google_reg_clip == True):
            D, U, V, conv_shape = full_svd(arr[0], in_shape)
            n_kern = Clip_OperatorNorm(D, U, V, conv_shape, 1)
            layer.set_weights([n_kern, arr[1]])
          if(google_reg_norm == True):
            sing = singular_values(arr[0], in_shape)
            n_kern = normalize_kern(arr[0], tf.math.reduce_max(sing), 1)
            layer.set_weights([n_kern, arr[1]])
          if(l2_norm == True):
            n_kern = normalize_kern(arr[0], l2_bound(arr[0]), 1)
            layer.set_weights([n_kern, arr[1]])
          if(linf_norm == True):
            n_kern = normalize_kern(arr[0], linf_bound(arr[0]), 1)
            layer.set_weights([n_kern, arr[1]])


  # Training
for bound in [1]:
  # Model
  acc_list = []
  bound_list = []
  model = Resnet32_c100()
  model.compile(optimizer='adam', loss='categorical_crossentropy', run_eagerly=True, metrics=['accuracy'])
  #o_path = "../experiments/convnet2_l2_norm"+str(bound)+".txt"
  #f = open(o_path, "x")
  start = time.time()
  for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    # Iterate over the batches of the augmented dataset.
    train_step(train_dataset)
 
    #TODO: Enable this when running anything apart from google reg clip
    # Iterate over the batches of the base dataset.
    train_step(train_dataset_base)

    _, train_acc = model.evaluate(x_train, y_train)
    _, test_acc = model.evaluate(x_test, y_test)
    acc_list.append([epoch, test_acc, train_acc])

    if (bound_logging):
      for lnum, layer in enumerate(model.layers):
        if(isinstance(layer, tf.keras.layers.Conv2D)):
          arr = layer.get_weights()
          in_shape = layer.input_shape[1:3]
          bound_list.append([epoch, lnum, l2_bound(arr[0]).numpy(),
            tf.math.reduce_max(singular_values(arr[0], in_shape)).numpy()])
        
    #if (epoch % 5 == 0):
      #print("Test accuracy at the end of epoch "+str(epoch)+" is "+str(test_acc))
  end = time.time()
  acc_list.append([end-start, -1, -1])
  #test_loss, test_accuracy = model.evaluate(x=x_test, y=y_test)
  #train_loss, train_accuracy = model.evaluate(x=x_train, y=y_train)
  # Print out the model accuracy 
  #print('\nTrain accuracy:', train_accuracy)
  #print('\nTest accuracy:', test_accuracy)
  #f.write("Test Accuracy: "+str(test_accuracy)+"\n")
  #f.write("Train Accuracy: "+str(train_accuracy)+"\n")
  #f.write("Time taken "+str(end-start)+"\n")
  #f.write("Epochs vs accuracy\n")
  #f.write(str(np.array(acc_list)))
  #if(bound_logging):
  #  f.write("\nBounds\n")
  #  f.write(str(np.array(bound_list)))
  #f.close()
  np.savetxt("../experiments/convnet2_c100_l2_acc.csv", np.array(acc_list), delimiter=",")
  if(bound_logging):
    np.savetxt("../experiments/convnet2_c100_l2_bounds.csv", np.array(bound_list), delimiter=",")
model.save("/home/gk/anirudh/experiments/models/convnet2_c100_l2")
