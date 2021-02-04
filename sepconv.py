import tensorflow as tf
import numpy as np


class SpaceDepthSepConv2(tf.keras.layers.Layer):
    def __init__(self, kern_size = 3, norm_flag=False, stride=1, **kwarg):
        if('name' in kwarg):
          super(SpaceDepthSepConv2, self).__init__(name = kwarg['name'])
        else:
          super(SpaceDepthSepConv2, self).__init__()
        self.u = self.add_weight(
            shape=([kwarg['out_channels'], kern_size]), initializer=tf.keras.initializers.glorot_normal, trainable=True
        )
        self.v = self.add_weight(
            shape=([kwarg['out_channels'], kern_size]), initializer=tf.keras.initializers.glorot_normal, trainable=True
        )
        self.w = self.add_weight(
            shape=([kwarg['out_channels'], kwarg['input_dim'][-1]]), initializer=tf.keras.initializers.glorot_normal, trainable=True
        ) 
        self.out_channels = tf.Variable(
            initial_value=tf.constant(kwarg['out_channels']), trainable=False
        )
        self.in_channels = tf.Variable(
            initial_value=tf.constant(kwarg['input_dim'][-1]), trainable=False
        )
        self.kern_size = tf.Variable(
            initial_value=tf.constant(kern_size), trainable=False
        )
        self.b = self.add_weight(
            shape=([kwarg['out_channels']]), initializer=tf.keras.initializers.zeros, trainable=True
        )
        self.in_shape = tf.Variable(
            initial_value=tf.constant(kwarg['input_dim']), trainable=False
        )
        self.norm_flag = tf.Variable(
            initial_value=tf.constant(norm_flag), trainable=False
        )
        self.stride = tf.Variable(
            initial_value=tf.constant(stride), trainable=False
        )

    def construct_kern(self):
        #Build kernel using u, v, w here
        #kern[i,j,k,l] = u[l,j]*w[l,k]*v[l,i]
        return tf.einsum('lj,lk, li->ijkl', self.u, self.w, self.v)

    def call(self, inputs):
        kern = self.construct_kern()
        return tf.keras.activations.relu(tf.nn.conv2d(inputs, kern, strides=self.stride, padding='SAME') + self.b)

    #Compute the bound as mentioned in the Design Doc
    def linf_bound(self):
        kern = self.construct_kern()
        return tf.math.reduce_max(tf.math.reduce_sum(tf.math.abs(kern), [0,1,2]))

    def l2_bound(self):
        kern = tf.cast(tf.transpose(self.construct_kern(), perm = [2,3,0,1]), dtype=tf.complex64)
        kern_hat = tf.math.square(tf.math.abs(tf.signal.fft2d(kern)))
        return tf.reduce_max(tf.math.sqrt(tf.math.reduce_sum(kern_hat, axis=[0,1])), axis=None)

    #Normalize in the L_inf - L_inf lipschitz sense
    def normalize_linf(self):
        self.div_kernel(self.linf_bound())

    def div_kernel(self, L):
        crL = tf.math.pow(L,1.0/3.0)
        self.u = tf.math.divide(self.u, crL)
        self.v = tf.math.divide(self.v, crL) 
        self.w = tf.math.divide(self.w, crL)

    def normalize_l2(self):
        self.div_kernel(self.l2_bound_exp())

    #Google paper inspired regularization
    def normalize_l2_op(self, clip_to):

        
        u_hat = tf.signal.fft2d(tf.cast(self.u, dtype = tf.complex64))
        u_hat_norm, _ = tf.linalg.normalize(u_hat, ord=2, axis=0)
        v_hat = tf.signal.fft2d(tf.cast(self.v, dtype = tf.complex64))
        v_hat_norm, _ = tf.linalg.normalize(v_hat, ord=2, axis=0)
        D, U, V = tf.linalg.svd(tf.cast(self.w, dtype = tf.complex64))
        D_clipped = tf.cast(tf.minimum(D, clip_to), tf.complex64)
        self.w = tf.math.real(tf.matmul(U, tf.matmul(tf.linalg.diag(D_clipped), V, adjoint_b=True)))
        self.u = tf.math.real(tf.signal.ifft2d(u_hat_norm))
        self.v = tf.math.real(tf.signal.ifft2d(v_hat_norm))
        


    def l2_bound_exp(self):
        u_hat = tf.signal.fft(tf.cast(self.u, dtype = tf.complex64))
        v_hat = tf.signal.fft(tf.cast(self.v, dtype = tf.complex64))
        w_out = tf.reduce_sum(self.w, axis=[1])
        temp = tf.einsum('li, lj, l-> lij', tf.math.square(tf.abs(u_hat)), tf.math.square(tf.math.abs(v_hat)), tf.math.square(w_out))
        freq_mat = tf.reduce_sum(temp, axis=[0])
        return tf.reduce_max(tf.math.sqrt(freq_mat), axis=None)

    def debug(self):
        #if (tf.norm(self.u, ord=2, axis=1).numpy().any() > 1):
        #   print("u out of norm bound")
        #if (tf.norm(self.v, ord=2, axis=1).numpy().any() > 1):
        #   print("v out of norm bound")
        #if (tf.norm(self.w, ord=np.inf, axis=1).numpy().any() > 1):
        #   print("w out of norm bound")
        print("L_inf Bound is "+str(self.linf_bound()))
        print("L_2 Bound is "+str(self.l2_bound_exp()))


class SpaceSepConv2(tf.keras.layers.Layer):
    def __init__(self, kern_size = 3, **kwarg):
        if(len(kwarg)>2):
          super(SpaceSepConv2, self).__init__(name = kwarg['name'])
        else:
          super(SpaceSepConv2, self).__init__()
        self.u = self.add_weight(
            shape=([kwarg['input_dim'][-1], kwarg['out_channels'], kern_size]), initializer=tf.keras.initializers.glorot_normal, trainable=True
        )
        self.v = self.add_weight(
            shape=([kwarg['input_dim'][-1], kwarg['out_channels'], kern_size]), initializer=tf.keras.initializers.glorot_normal, trainable=True
        )
        self.out_channels = tf.Variable(
            initial_value=tf.constant(kwarg['out_channels']), trainable=False
        )
        self.in_channels = tf.Variable(
            initial_value=tf.constant(kwarg['input_dim'][-1]), trainable=False
        )
        self.kern_size = tf.Variable(
            initial_value=tf.constant(kern_size), trainable=False
        )
        self.b = self.add_weight(
            shape=(kwarg['input_dim'][-3:-1]+[kwarg['out_channels']]), initializer=tf.keras.initializers.zeros, trainable=True
        )

    def construct_kern(self):
        #Build kernel using u, v here
        #kern[i,j,k,l] = u[k,l,j]*w[k,l,i]*v[l,i]
        return tf.einsum('klj,kli->ijkl', self.u, self.v)

    def call(self, inputs):    
        kern = self.construct_kern()
        return tf.keras.activations.relu(tf.nn.conv2d(inputs, kern, [1,1,1,1], padding='SAME') + self.b)
    #Compute the bound as mentioned in the Design Doc
    def compute_bound(self):
        kern = self.construct_kern()
        return tf.math.reduce_max(tf.math.reduce_sum(tf.math.abs(kern), [0,1,2]))
    #Normalize in the L_inf - L_inf lipschitz sense
    def normalize_layer(self):
        L = self.compute_bound()
        srL = tf.math.pow(L,1.0/2.0)
        self.u = tf.math.divide(self.u, srL)
        self.v = tf.math.divide(self.v, srL)
