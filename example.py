import tensorflow as tf
from net_sys import nnetSystem
from model_designs import convnet_conv3

#Temporary fix for evaluation, we shouldn't need to access the data outside nnetSystem! 
(x_train, y_train),(x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype(float)
x_test = x_test.astype(float)

#Build the required model object, you can create any tensorflow.models.Sequential model here
model = convnet_conv3()

#Initialize our system with model in question. We can also specify the optimizer, loss function etc.
proj = nnetSystem(model)

#begin training
proj.training_loop(linf_norm=False)

# Evaluate the model performance
test_loss, test_acc = proj.model.evaluate(x=x_test, y=y_test)

# Print out the model accuracy 
print('\nTest accuracy:', test_acc)

