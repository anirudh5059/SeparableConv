import tensorflow as tf
from sepconv import SpaceDepthSepConv2
import numpy as np

class nnetSystem():
  def __init__(self, model, batch_size = 256, data = 'cifar10', opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9), loss_fn = tf.keras.losses.CategoricalCrossentropy()):
    super(nnetSystem, self).__init__()
    self.model = tf.keras.models.Sequential()
    self.batch_size = batch_size
    if data == 'cifar10':
      (self.x_train, self.y_train),(self.x_test, self.y_test) = tf.keras.datasets.cifar10.load_data()
      self.convert_categorical(10)
    if data == 'cifar100':
      (self.x_train, self.y_train),(self.x_test, self.y_test) = tf.keras.datasets.cifar100.load_data()
      self.convert_categorical(100)
    self.prep_data()
    self.opt = opt
    self.loss_fn = loss_fn
    self.model = model
    
  #Funcion to load the CIFAR10 dataset and create a data structure which allows iterating over batches.
  def prep_data(self):
    self.x_train = self.x_train.astype(float)
    self.x_test = self.x_test.astype(float)
    mean = np.mean(self.x_train,axis=(0,1,2,3))
    std = np.std(self.x_train,axis=(0,1,2,3))
    self.x_train = (self.x_train-mean)/(std+1e-7)
    self.x_test = (self.x_test-mean)/(std+1e-7)
    
  def convert_categorical(self, num_classes):
    self.y_train = tf.keras.utils.to_categorical(self.y_train,num_classes)
    self.y_test = tf.keras.utils.to_categorical(self.y_test,num_classes)

  #Prepare iterable data structure
  def data_iterable(self):
    train_dataset = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
    return train_dataset.shuffle(buffer_size=1024).batch(self.batch_size)

  def data_iterable_aug(self):
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
    datagen.fit(self.x_train)
    return datagen.flow(self.x_train, self.y_train, batch_size=self.batch_size)

  def evaluate(self):
    # Evaluate the model performance
    test_loss, test_acc = self.model.evaluate(x=self.x_test, y=self.y_test)
    train_loss, train_accuracy = self.model.evaluate(x=self.x_train, y=self.y_train)
    # Print out the model accuracy 
    print('\nTrain accuracy:', train_accuracy)
    print('\nTest accuracy:', test_acc)

  def training_loop(self, epochs = 10, linf_norm = False, l2_norm=False, k = 15, data_aug = False, google_reg=False):
    
    if data_aug == True:
      train_dataset = self.data_iterable_aug()
    else:
      train_dataset = self.data_iterable()

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
                logits = self.model(x_batch_train, training=True)  # Logits for this minibatch

                # Compute the loss value for this minibatch.
                loss_value = self.loss_fn(y_batch_train, logits)

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, self.model.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            self.opt.apply_gradients(zip(grads, self.model.trainable_weights))

            #Regularization and logging
            if(step > len(self.x_train)/self.batch_size):
              break
            if step % k == 0:
              for layer in self.model.layers:
                if(isinstance(layer, SpaceDepthSepConv2) and (layer.norm_flag == True)):
                  #print("Got here")
                  #layer.debug()
                  if(linf_norm == True):
                    layer.normalize_linf()
                  if(l2_norm == True):
                    layer.normalize_l2()
                  if(google_reg == True):
                    layer.normalize_google()
              print(
                  "Training loss (for one batch) at step %d: %.4f"
                  % (step, float(loss_value))
              )
              #print("Seen so far: %s samples" % ((step + 1) * self.batch_size))
