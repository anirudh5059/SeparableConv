import tensorflow as tf
from sepconv import SpaceSepConv2
from sepconv import SpaceDepthSepConv2
from sepconv import DepthSepConv2
from conv3 import Conv3
from conv3 import FullConv3

def convnet1(): 
    
    kern = 3
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(64, kern, input_shape=[32,32,3],
        padding='same', activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(128, kern, padding='same',
        activation='relu', kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(256, kern, padding='same',
        activation='relu', kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(512, kern, padding='same',
        activation='relu', kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu,
        kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax,
        kernel_regularizer=tf.keras.regularizers.l2(5e-4)))

    return model

def convnet2(): 
    
    kern = 3
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(64, kern, input_shape=[32,32,3],
        padding='same', activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(64, kern, padding='same',
        activation='relu', kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Conv2D(128, kern, padding='same',
        activation='relu', kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(128, kern, padding='same',
        activation='relu', kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Conv2D(256, kern, padding='same',
        activation='relu', kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(256, kern, padding='same',
        activation='relu', kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Conv2D(512, kern, padding='same',
        activation='relu', kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(512, kern, padding='same',
        activation='relu', kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu,
        kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax,
        kernel_regularizer=tf.keras.regularizers.l2(5e-4)))


    return model

def convnet2_c100(): 
    
    kern = 3
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(64, kern, input_shape=[32,32,3],
        padding='same', activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(64, kern, padding='same',
        activation='relu', kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Conv2D(128, kern, padding='same',
        activation='relu', kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(128, kern, padding='same',
        activation='relu', kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Conv2D(256, kern, padding='same',
        activation='relu', kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(256, kern, padding='same',
        activation='relu', kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Conv2D(512, kern, padding='same',
        activation='relu', kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(512, kern, padding='same',
        activation='relu', kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu,
        kernel_regularizer=tf.keras.regularizers.l2(5e-4)))

    model.add(tf.keras.layers.Dense(100, activation=tf.nn.softmax,
        kernel_regularizer=tf.keras.regularizers.l2(5e-4)))

    return model

def convnet_conv3(): 
    
    kern = 3
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(64, kern, padding='same',
        activation='relu', kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(128, kern, padding='same',
        activation='relu', kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(Conv3(input_dim=[8,8,128], out_mult=2))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(FullConv3(input_dim=[4,4,256], out_mult=2))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu,
        kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu,
        kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu,
        kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(1024, activation=tf.nn.relu,
        kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax,
        kernel_regularizer=tf.keras.regularizers.l2(5e-4)))

    return model

def spacesepconv1():

    kern = 3
    model = tf.keras.models.Sequential()

    model.add(SpaceSepConv2(input_dim=[32,32,3], out_channels=64,
        norm_flag=True))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(SpaceSepConv2(input_dim=[16,16,64], out_channels=128,
        norm_flag=True))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(SpaceSepConv2(input_dim=[8,8,128], out_channels=256,
        norm_flag=True))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(SpaceSepConv2(input_dim=[4,4,256], out_channels=512,
        norm_flag=True))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu,
        kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax,
        kernel_regularizer=tf.keras.regularizers.l2(0)))

    
    return model

def depthsepconv1():

    kern = 3
    model = tf.keras.models.Sequential()

    model.add(DepthSepConv2(input_dim=[32,32,3], out_channels=64,
        norm_flag=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(DepthSepConv2(input_dim=[32,32,64], out_channels=64,
        norm_flag=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(DepthSepConv2(input_dim=[16,16,64], out_channels=128,
        norm_flag=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(DepthSepConv2(input_dim=[16,16,128], out_channels=128,
        norm_flag=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(DepthSepConv2(input_dim=[8,8,128], out_channels=256,
        norm_flag=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(DepthSepConv2(input_dim=[8,8,256], out_channels=256,
        norm_flag=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(DepthSepConv2(input_dim=[4,4,256], out_channels=512,
        norm_flag=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(DepthSepConv2(input_dim=[4,4,512], out_channels=512,
        norm_flag=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu,
        kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax,
        kernel_regularizer=tf.keras.regularizers.l2(0)))

    return model

def spacedepthsepconv2():

    kern = 3
    model = tf.keras.models.Sequential()

    model.add(SpaceDepthSepConv2(input_dim=[32,32,3], out_channels=64,
        norm_flag=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(SpaceDepthSepConv2(input_dim=[32,32,64], out_channels=64,
        norm_flag=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(SpaceDepthSepConv2(input_dim=[16,16,64], out_channels=128,
        norm_flag=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(SpaceDepthSepConv2(input_dim=[16,16,128], out_channels=128,
        norm_flag=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(SpaceDepthSepConv2(input_dim=[8,8,128], out_channels=256,
        norm_flag=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(SpaceDepthSepConv2(input_dim=[8,8,256], out_channels=256,
        norm_flag=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(SpaceDepthSepConv2(input_dim=[4,4,256], out_channels=512,
        norm_flag=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(SpaceDepthSepConv2(input_dim=[4,4,512], out_channels=512,
        norm_flag=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu,
        kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax,
        kernel_regularizer=tf.keras.regularizers.l2(0)))


    return model

def spacedepthsepconv3():

    kern = 3
    model = tf.keras.models.Sequential()

    model.add(SpaceDepthSepConv2(input_dim=[32,32,3], out_channels=128,
        norm_flag=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(SpaceDepthSepConv2(input_dim=[32,32,64], out_channels=128,
        norm_flag=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(SpaceDepthSepConv2(input_dim=[16,16,128], out_channels=256,
        norm_flag=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(SpaceDepthSepConv2(input_dim=[16,16,256], out_channels=256,
        norm_flag=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(SpaceDepthSepConv2(input_dim=[8,8,256], out_channels=512,
        norm_flag=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(SpaceDepthSepConv2(input_dim=[8,8,512], out_channels=512,
        norm_flag=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(SpaceDepthSepConv2(input_dim=[4,4,512], out_channels=1024,
        norm_flag=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(SpaceDepthSepConv2(input_dim=[4,4,1024], out_channels=1024,
        norm_flag=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1024, activation=tf.nn.relu,
        kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu,
        kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax,
        kernel_regularizer=tf.keras.regularizers.l2(0)))

    return model


def vgg_spacedepth():

    kern = 3
    model = tf.keras.models.Sequential()

    model.add(SpaceDepthSepConv2(input_dim=[32,32,3], out_channels=64))
    model.add(SpaceDepthSepConv2(input_dim=[32,32,64], out_channels=64,
        norm_flag=True))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(SpaceDepthSepConv2(input_dim=[16,16,64], out_channels=128))
    model.add(SpaceDepthSepConv2(input_dim=[16,16,128], out_channels=128,
        norm_flag=True))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(SpaceDepthSepConv2(input_dim=[8,8,128], out_channels=256))
    model.add(SpaceDepthSepConv2(input_dim=[8,8,256], out_channels=256,
        norm_flag=True))
    model.add(SpaceDepthSepConv2(input_dim=[8,8,256], out_channels=256))
    model.add(SpaceDepthSepConv2(input_dim=[8,8,256], out_channels=256,
        norm_flag=True))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(SpaceDepthSepConv2(input_dim=[4,4,256], out_channels=512))
    model.add(SpaceDepthSepConv2(input_dim=[4,4,512], out_channels=512,
        norm_flag=True))
    model.add(SpaceDepthSepConv2(input_dim=[4,4,512], out_channels=512))
    model.add(SpaceDepthSepConv2(input_dim=[4,4,512], out_channels=512,
        norm_flag=True))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(SpaceDepthSepConv2(input_dim=[2,2,512], out_channels=512))
    model.add(SpaceDepthSepConv2(input_dim=[2,2,512], out_channels=512,
        norm_flag=True))
    model.add(SpaceDepthSepConv2(input_dim=[2,2,512], out_channels=512))
    model.add(SpaceDepthSepConv2(input_dim=[2,2,512], out_channels=512,
        norm_flag=True))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(4096, activation=tf.nn.relu,
        kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(4096, activation=tf.nn.relu,
        kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(1024, activation=tf.nn.relu,
        kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax,
        kernel_regularizer=tf.keras.regularizers.l2(5e-4)))

    return model

def vgg11():

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(64, kern, activation='relu',
        padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(128, kern, activation='relu',
        padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(256, kern, activation='relu',
        padding='same'))
    model.add(tf.keras.layers.Conv2D(256, kern, activation='relu',
        padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(512, kern, activation='relu',
        padding='same'))
    model.add(tf.keras.layers.Conv2D(512, kern, activation='relu',
        padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(512, kern, activation='relu',
        padding='same'))
    model.add(tf.keras.layers.Conv2D(512, kern, activation='relu',
        padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(4096, activation=tf.nn.relu,
        kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(4096, activation=tf.nn.relu,
        kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(1024, activation=tf.nn.relu,
        kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(100, activation=tf.nn.softmax,
        kernel_regularizer=tf.keras.regularizers.l2(5e-4)))

    return model

def vgg11_spacedepth():

    model = tf.keras.models.Sequential()

    model.add(SpaceDepthSepConv2(input_dim=[32,32,3], out_channels=64))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(SpaceDepthSepConv2(input_dim=[16,16,64], out_channels=128))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(SpaceDepthSepConv2(input_dim=[8,8,128], out_channels=256))
    model.add(SpaceDepthSepConv2(input_dim=[8,8,256], out_channels=256))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(SpaceDepthSepConv2(input_dim=[4,4,256], out_channels=512))
    model.add(SpaceDepthSepConv2(input_dim=[4,4,512], out_channels=512))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(SpaceDepthSepConv2(input_dim=[2,2,512], out_channels=512))
    model.add(SpaceDepthSepConv2(input_dim=[2,2,512], out_channels=512))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(4096, activation=tf.nn.relu,
        kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(4096, activation=tf.nn.relu,
        kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(1024, activation=tf.nn.relu,
        kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(100, activation=tf.nn.softmax,
        kernel_regularizer=tf.keras.regularizers.l2(5e-4)))


    return model

def convnet_c100():
    
    kern = 3
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(64, kern, padding='same',
        activation='relu', kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(128, kern, padding='same',
        activation='relu', kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(256, kern, padding='same',
        activation='relu', kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(512, kern, padding='same',
        activation='relu', kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu,
        kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(100, activation=tf.nn.softmax,
        kernel_regularizer=tf.keras.regularizers.l2(5e-4)))

    return model

def spacedepthsepconv_c100():

    kern = 3
    model = tf.keras.models.Sequential()

    model.add(SpaceDepthSepConv2(input_dim=[32,32,3], out_channels=64,
        norm_flag=True))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(SpaceDepthSepConv2(input_dim=[16,16,64], out_channels=128,
        norm_flag=True))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(SpaceDepthSepConv2(input_dim=[8,8,128], out_channels=256,
        norm_flag=True))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(SpaceDepthSepConv2(input_dim=[4,4,256], out_channels=512,
        norm_flag=True))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu,
        kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(100, activation=tf.nn.softmax,
        kernel_regularizer=tf.keras.regularizers.l2(5e-4)))

    return model

def spacesepconv_c100():

    kern = 3
    model = tf.keras.models.Sequential()

    model.add(SpaceSepConv2(input_dim=[32,32,3], out_channels=64))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(SpaceSepConv2(input_dim=[16,16,64], out_channels=128))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(SpaceSepConv2(input_dim=[8,8,128], out_channels=256))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1024, activation=tf.nn.relu,
        kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(100, activation=tf.nn.softmax,
        kernel_regularizer=tf.keras.regularizers.l2(5e-4)))

    return model

def vgg_c100():

    kern = 3
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(64, kern, activation='relu',
        padding='same'))
    model.add(tf.keras.layers.Conv2D(64, kern, activation='relu',
        padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(128, kern, activation='relu',
        padding='same'))
    model.add(tf.keras.layers.Conv2D(128, kern, activation='relu',
        padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(256, kern, activation='relu',
        padding='same'))
    model.add(tf.keras.layers.Conv2D(256, kern, activation='relu',
        padding='same'))
    model.add(tf.keras.layers.Conv2D(256, kern, activation='relu',
        padding='same'))
    model.add(tf.keras.layers.Conv2D(256, kern, activation='relu',
        padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(512, kern, activation='relu',
        padding='same'))
    model.add(tf.keras.layers.Conv2D(512, kern, activation='relu',
        padding='same'))
    model.add(tf.keras.layers.Conv2D(512, kern, activation='relu',
        padding='same'))
    model.add(tf.keras.layers.Conv2D(512, kern, activation='relu',
        padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(512, kern, activation='relu',
        padding='same'))
    model.add(tf.keras.layers.Conv2D(512, kern, activation='relu',
        padding='same'))
    model.add(tf.keras.layers.Conv2D(512, kern, activation='relu',
        padding='same'))
    model.add(tf.keras.layers.Conv2D(512, kern, activation='relu',
        padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(4096, activation=tf.nn.relu,
        kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(4096, activation=tf.nn.relu,
        kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(1024, activation=tf.nn.relu,
        kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(100, activation=tf.nn.softmax,
        kernel_regularizer=tf.keras.regularizers.l2(5e-4)))

    return model

def vgg_spacedepth_c100():

    kern = 3
    model = tf.keras.models.Sequential()

    model.add(SpaceDepthSepConv2(input_dim=[32,32,3], out_channels=64))
    model.add(SpaceDepthSepConv2(input_dim=[32,32,64], out_channels=64,
        norm_flag=True))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(SpaceDepthSepConv2(input_dim=[16,16,64], out_channels=128))
    model.add(SpaceDepthSepConv2(input_dim=[16,16,128], out_channels=128,
        norm_flag=True))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(SpaceDepthSepConv2(input_dim=[8,8,128], out_channels=256))
    model.add(SpaceDepthSepConv2(input_dim=[8,8,256], out_channels=256,
        norm_flag=True))
    model.add(SpaceDepthSepConv2(input_dim=[8,8,256], out_channels=256))
    model.add(SpaceDepthSepConv2(input_dim=[8,8,256], out_channels=256,
        norm_flag=True))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(SpaceDepthSepConv2(input_dim=[4,4,256], out_channels=512))
    model.add(SpaceDepthSepConv2(input_dim=[4,4,512], out_channels=512,
        norm_flag=True))
    model.add(SpaceDepthSepConv2(input_dim=[4,4,512], out_channels=512))
    model.add(SpaceDepthSepConv2(input_dim=[4,4,512], out_channels=512,
        norm_flag=True))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(SpaceDepthSepConv2(input_dim=[2,2,512], out_channels=512))
    model.add(SpaceDepthSepConv2(input_dim=[2,2,512], out_channels=512,
        norm_flag=True))
    model.add(SpaceDepthSepConv2(input_dim=[2,2,512], out_channels=512))
    model.add(SpaceDepthSepConv2(input_dim=[2,2,512], out_channels=512,
        norm_flag=True))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(4096, activation=tf.nn.relu,
        kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(4096, activation=tf.nn.relu,
        kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(1024, activation=tf.nn.relu,
        kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(100, activation=tf.nn.softmax, 
        kernel_regularizer=tf.keras.regularizers.l2(5e-4)))

    return model


def simplenet_spacedepth():

    baseMapNum = 32
    weight_decay = 1e-4
    model = tf.keras.models.Sequential()

    model.add(SpaceDepthSepConv2(input_dim=[32,32,3],
        out_channels=baseMapNum, norm_flag=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(SpaceDepthSepConv2(input_dim=[32,32,baseMapNum],
        out_channels=baseMapNum, norm_flag=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    
    model.add(SpaceDepthSepConv2(input_dim=[16,16,baseMapNum],
        out_channels=2*baseMapNum, norm_flag=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(SpaceDepthSepConv2(input_dim=[16,16,2*baseMapNum],
        out_channels=2*baseMapNum, norm_flag=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(SpaceDepthSepConv2(input_dim=[8,8,2*baseMapNum],
        out_channels=4*baseMapNum, norm_flag=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(SpaceDepthSepConv2(input_dim=[8,8,4*baseMapNum],
        out_channels=4*baseMapNum, norm_flag=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.4))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    return model

def simplenet():

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(baseMapNum, (3,3), padding='same',
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
        activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(baseMapNum, (3,3), padding='same',
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
        activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv2D(2*baseMapNum, (3,3), padding='same',
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
        activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(2*baseMapNum, (3,3), padding='same',
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
        activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(4*baseMapNum, (3,3), padding='same',
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
        activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(4*baseMapNum, (3,3), padding='same',
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
        activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.4))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    return model


def allconvnet():

    kern = 3
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(96, kern, padding='same',
        activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3)))
    model.add(tf.keras.layers.Conv2D(96, kern, padding='same',
        activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3)))
    model.add(tf.keras.layers.Conv2D(96, kern, padding='same',
        activation='relu', strides=2,
        kernel_regularizer=tf.keras.regularizers.l2(1e-3)))

    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Conv2D(192, kern, padding='same',
        activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3)))
    model.add(tf.keras.layers.Conv2D(192, kern, padding='same',
        activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3)))
    model.add(tf.keras.layers.Conv2D(192, kern, padding='same',
        activation='relu', strides=2,
        kernel_regularizer=tf.keras.regularizers.l2(1e-3)))

    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Conv2D(192, kern, padding='same',
        activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3)))
    model.add(tf.keras.layers.Conv2D(192, 1, padding='same', activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(1e-3)))
    model.add(tf.keras.layers.Conv2D(10, 1, padding='same', activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(1e-3)))

    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(10, activation='softmax',
        kernel_regularizer=tf.keras.regularizers.l2(1e-3)))

    return model

def allconvnet_BN():

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(96, kern, padding='same',
        activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3)))
    model.add(tf.keras.layers.Conv2D(96, kern, padding='same',
        activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3)))
    model.add(tf.keras.layers.Conv2D(96, kern, padding='same',
        activation='relu', strides=2,
        kernel_regularizer=tf.keras.regularizers.l2(1e-3)))
    model.add(BatchNormalization())

    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Conv2D(192, kern, padding='same',
        activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3)))
    model.add(tf.keras.layers.Conv2D(192, kern, padding='same',
        activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3)))
    model.add(tf.keras.layers.Conv2D(192, kern, padding='same',
        activation='relu', strides=2,
        kernel_regularizer=tf.keras.regularizers.l2(1e-3)))
    model.add(BatchNormalization())

    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Conv2D(192, kern, padding='same', 
        activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3)))
    model.add(tf.keras.layers.Conv2D(192, 1, padding='same',
        activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3)))
    model.add(tf.keras.layers.Conv2D(10, 1, padding='same',
        activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3)))
    model.add(BatchNormalization())

    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(10, activation='softmax',
        kernel_regularizer=tf.keras.regularizers.l2(1e-3)))

    return model

 
# Helper function to build the resnet model.
def build_residual_blocks(inp, filters, reps, subsample=False):
    
    # We need a list class which can be grown dynamically.
    class GrowingList(list):
        def __setitem__(self, index, value):
                if index >= len(self):
                        self.extend([None]*(index + 1 - len(self)))
                list.__setitem__(self, index, value)

    stride = 2 if subsample else 1

    # Initialize an empty list of the requisite layers here.

    # First group of BatchNormalization, activation and convolution layers.
    bn1 = GrowingList()
    act1 = GrowingList()
    c1 = GrowingList()

    # Second group of BatchNormalization, activation and convolution layers.
    bn2 = GrowingList()                                                             
    act2 = GrowingList()
    c2 = GrowingList()

    # Adder layer.
    add = GrowingList()
    
    # Initialize the first set of layers.
    bn1[0] = tf.keras.layers.BatchNormalization()(inp)
    act1[0] = tf.keras.layers.ReLU()(bn1[0])

    # If subsampling at beginning, need a stride of 2 only for 1st conv layer.
    c1[0] = tf.keras.layers.Conv2D(filters, 3, padding='same', 
        strides=stride)(act1[0])
    
    bn2[0] = tf.keras.layers.BatchNormalization()(c1[0])
    act2[0] = tf.keras.layers.ReLU()(bn2[0])
    c2[0] = tf.keras.layers.Conv2D(filters, 3, padding='same')(act2[0])
    
    # In case of subsampling, additional processing needed to make the 
    # dimensions of the two summands of the adder match up. We use maxpooling
    # for the spatial dimensions and zero padding for the filter dimension.
    if subsample:
        in_pooled = tf.keras.layers.MaxPooling2D()(inp)
        in_padded = tf.pad(in_pooled, paddings=[
                                                [0, 0],
                                                [0, 0],
                                                [0, 0],
                                                [0, filters//2]
                                                ])
        add[0] = tf.keras.layers.Add()([in_padded, c2[0]])
    else:
        add[0] = tf.keras.layers.Add()([inp, c2[0]])

    # For the layer layers in the residual block, follow same hierarchy.
    for i in range(1, reps):
        bn1[i] = tf.keras.layers.BatchNormalization()(add[i-1])
        act1[i] = tf.keras.layers.ReLU()(bn1[i])
        c1[i] = tf.keras.layers.Conv2D(filters, 3, padding='same')(act1[i])

        bn2[i] = tf.keras.layers.BatchNormalization()(c1[i])
        act2[i] = tf.keras.layers.ReLU()(bn2[i])
        c2[i] = tf.keras.layers.Conv2D(filters, 3, padding='same')(act2[i])
        
        add[i] = tf.keras.layers.Add()([add[i-1], c2[i]])

    return add[reps-1]
    
# 32 layer Resnet model for 10 class classification.
def Resnet32():

    inp = tf.keras.Input(shape=[32,32,3])
    in_conv = tf.keras.layers.Conv2D(64, 3, padding='same')(inp)
    res_64 = build_residual_blocks(in_conv, 64, 3)
    res_128 = build_residual_blocks(res_64, 128, 4, subsample=True)
    res_256 = build_residual_blocks(res_128, 256, 6, subsample=True)
    res_512 = build_residual_blocks(res_256, 512, 4, subsample=True)
    avg = tf.keras.layers.GlobalAveragePooling2D()(res_512)
    out = tf.keras.layers.Dense(10, activation='softmax')(avg)
    model = tf.keras.models.Model(inputs=inp, outputs=out)

    return model

# ResNet32 for 100 class classification.
def Resnet32_c100():

    inp = tf.keras.Input(shape=[32,32,3])
    in_conv = tf.keras.layers.Conv2D(64, 3, padding='same')(inp)
    res_64 = build_residual_blocks(in_conv, 64, 3)
    res_128 = build_residual_blocks(res_64, 128, 4, subsample=True)
    res_256 = build_residual_blocks(res_128, 256, 6, subsample=True)
    res_512 = build_residual_blocks(res_256, 512, 4, subsample=True)
    avg = tf.keras.layers.GlobalAveragePooling2D()(res_512)
    out = tf.keras.layers.Dense(100, activation='softmax')(avg)
    model = tf.keras.models.Model(inputs=inp, outputs=out)

    return model
