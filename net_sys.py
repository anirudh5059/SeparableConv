import tensorflow as tf
from sepconv import SpaceDepthSepConv2
import numpy as np
from sepconv import linf_bound
from sepconv import l2_bound
from sepconv import singular_values
from sepconv import Normalize_kern

class nnetSystem():
    '''
        Class to encapsulate model training and evaluation

        Attributes
        ----------
            model(tf.keras.Model): keras model architecture used for training.
            batch_size(int):  Number of examples per minibatch.
            data(string): label indicating to the class which data to use.
            opt(tf.keras.optimizers.Optimizer): optimizer for weight update.
            loss(tf.keras.losses.Loss): loss function for training.
            x_train(np.array): Training data matrix
            y_train(np.array): Labels corresponding to the train data matrix.
            x_test(np.array): Testing data matrix
            y_test(np.array): Labels corresponding to the test data matrix.

        Methods
        -------
            prep_data: Scale and normalize the loaded data.
            convert_categorical: Convert the labels to one hot vectors.
            data_iterable: Create and return the base data generator.
            data_iterable_aug: Create and return the augmented data generator.
            evaluate: Evaluate the current model on test data.
            training_loop: Implement the training loop.
            
    '''
    def __init__(self, model, batch_size=256, data='cifar10',
        opt=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        loss_fn=tf.keras.losses.CategoricalCrossentropy()):
        super(nnetSystem, self).__init__()
        self.model = tf.keras.models.Sequential()
        self.batch_size = batch_size
        if data == 'cifar10':
            (self.x_train, self.y_train),(self.x_test, self.y_test) = 
        tf.keras.datasets.cifar10.load_data()
            self.convert_categorical(10)
        if data == 'cifar100':
            (self.x_train, self.y_train),(self.x_test, self.y_test) =
        tf.keras.datasets.cifar100.load_data()
            self.convert_categorical(100)
        self.prep_data()
        self.opt = opt
        self.loss_fn = loss_fn
        self.model = model
        
    def prep_data(self):
    '''
        Scale and normalize the loaded data.
    '''
        self.x_train = self.x_train.astype(float)
        self.x_test = self.x_test.astype(float)
        mean = np.mean(self.x_train,axis=(0,1,2,3))
        std = np.std(self.x_train,axis=(0,1,2,3))
        self.x_train = (self.x_train-mean)/(std+1e-7)
        self.x_test = (self.x_test-mean)/(std+1e-7)
        
    def convert_categorical(self, num_classes):
    '''
        Convert the labels to one hot vectors.
    '''
        self.y_train = tf.keras.utils.to_categorical(self.y_train,num_classes)
        self.y_test = tf.keras.utils.to_categorical(self.y_test,num_classes)

    def data_iterable(self):
    '''
        Create and return the base data generator.
    '''
        train_dataset = tf.data.Dataset.from_tensor_slices((self.x_train,
        self.y_train))
        return train_dataset.shuffle(buffer_size=1024).batch(self.batch_size)

    def data_iterable_aug(self):
    '''
        Create the return the augmented data generator.
    '''
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
        return datagen.flow(self.x_train, self.y_train,
            batch_size=self.batch_size)

    def evaluate(self):
    '''
        Evaluate the model performance
    '''
        test_loss, test_acc = self.model.evaluate(x=self.x_test, y=self.y_test)
        train_loss, train_accuracy = self.model.evaluate(x=self.x_train, 
                y=self.y_train)
        # Print out the model accuracy 
        print('\nTrain accuracy:', train_accuracy)
        print('\nTest accuracy:', test_acc)

    def training_loop(self, epochs=10, linf_norm=False, l2_norm=False, k=15,
        data_aug=False, google_reg=False):
    '''
        Function implementing the training loop

        Inputs
        ------
            epochs(int): Iterate over all the data this many times.
            linf_norm(bool): flag to perform linf-linf lip reg.
            l2_norm(bool): flag to perform l2-l2 lip reg.
            google_reg(bool): flag to perform linf-linf lip reg.
            data_aug(bool): flag to use augmented dataset..
            k(int): Number of minibatch updates after which we regularize.
    '''
        
        if data_aug == True:
            train_dataset = self.data_iterable_aug()
        else:
            train_dataset = self.data_iterable()

        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))

            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train)
                in enumerate(train_dataset):


                    # Open a GradientTape to record the operations run
                # during the forward pass, which enables auto-differentiation.
                with tf.GradientTape() as tape:

                    # Run the forward pass of the layer.
                    # The operations that the layer applies
                    # to its inputs are going to be recorded
                    # on the GradientTape.

                    # Logits for this minibatch
                    logits = self.model(x_batch_train, training=True) 

                    # Compute the loss value for this minibatch.
                    loss_value = self.loss_fn(y_batch_train, logits)

                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the
                # loss.
                grads = tape.gradient(loss_value, self.model.trainable_weights)

                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.
                self.opt.apply_gradients(zip(grads,
                    self.model.trainable_weights))

                #Regularization and logging
                if(step > len(self.x_train)/self.batch_size):
                    break
                if step % k == 0:
                    for layer in self.model.layers:
                        # If custom convolutions:
                        if(
                            (isinstance(layer, SpaceDepthSepConv2)
                                or isinstance(layer, SpaceSepConv2) 
                                or isinstance(layer, DepthSepConv2))
                            and
                            (layer.norm_flag == True)
                        ):
                            if(linf_norm == True):
                                bound = linf_bound(layer.construct_kern())
                            if(l2_norm == True):
                                bound = l2_bound(layer.construct_kern())
                            if(google_reg == True):
                                bound = singular_values(layer.construct_kern())
                            layer.div_kern(bound)

                        # For regular convolutions:
                        if(isinstance(layer, tf.keras.layers.Conv2D)):
                            arr = layer.get_weights()
                            in_shape = layer.input_shape[1:3]
                            if(google_reg_ == True):
                                sing = singular_values(arr[0], in_shape)
                                n_kern = normalize_kern(arr[0],
                                    tf.math.reduce_max(sing), 1)
                                layer.set_weights([n_kern, arr[1]])
                            if(l2_norm == True):
                                n_kern = normalize_kern(arr[0],
                                    l2_bound(arr[0]),1)
                                layer.set_weights([n_kern, arr[1]])
                            if(linf_norm == True):
                                n_kern = normalize_kern(arr[0],
                                    linf_bound(arr[0]), 1)
                                layer.set_weights([n_kern, arr[1]])
                    print("Training loss (for one batch) at step %d: %.4f"
                        % (step, float(loss_value)))
