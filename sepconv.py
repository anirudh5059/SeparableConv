import tensorflow as tf
import numpy as np

# Functions taken straight from the google paper code
def prep_svd(conv, inp_shape):
  conv_tr = tf.cast(tf.transpose(conv, perm=[2, 3, 0, 1]), tf.complex64)
  conv_shape = conv.shape
  padding = tf.constant([[0, 0], [0, 0],
                         [0, inp_shape[0] - conv_shape[0]],
                         [0, inp_shape[1] - conv_shape[1]]])
  return tf.signal.fft2d(tf.pad(conv_tr, padding))

def singular_values(conv, inp_shape):
  transform_coeff = prep_svd(conv, inp_shape)
  singular_values = tf.linalg.svd(tf.transpose(transform_coeff, perm = [2, 3, 0, 1]),
                           compute_uv=False)
  return singular_values

def full_svd(conv, inp_shape):
  conv_shape = conv.shape
  transform_coeff = prep_svd(conv, inp_shape)
  D,U,V = tf.linalg.svd(tf.transpose(transform_coeff, perm = [2, 3, 0, 1]))
  return D, U, V, conv_shape

def Clip_OperatorNorm(D, U, V, conv_shape, clip_to):

  D_clipped = tf.cast(tf.minimum(D, clip_to), tf.complex64)
  clipped_coeff = tf.linalg.matmul(U, tf.linalg.matmul(tf.linalg.diag(D_clipped),
                                         V, adjoint_b=True))
  clipped_conv_padded = tf.math.real(tf.signal.ifft2d(
      tf.transpose(clipped_coeff, perm=[2, 3, 0, 1])))
  return tf.slice(tf.transpose(clipped_conv_padded, perm=[2, 3, 0, 1]),
                  [0] * len(conv_shape), conv_shape)

def Normalize_kern(kern, L, div):
  return tf.math.divide(kern, L/div)

def l2_bound(kern):
  kern_t = tf.cast(tf.transpose(kern, perm = [2,3,0,1]), dtype=tf.complex64)
  kern_hat = tf.math.square(tf.math.abs(tf.signal.fft2d(kern_t)))
  return tf.reduce_max(tf.math.sqrt(tf.math.reduce_sum(kern_hat, axis=[0,1])), axis=None)

def linf_bound(kern):
  return tf.math.reduce_max(tf.math.reduce_sum(tf.math.abs(kern), [0,1,2]))


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
            initial_value=tf.constant(kwarg['input_dim'][:-1]), trainable=False
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


    #Normalize in the L_inf - L_inf lipschitz sense
    def normalize_linf(self):
        self.div_kernel(self.linf_bound())

    def normalize_google(self, norm_bound):
      kern = self.construct_kern()
      D, U, V, _ =  SVD_Conv_Tensor(kern, self.in_shape.numpy())
      L = tf.math.reduce_max(D)
      self.div_kernel(L/norm_bound)
      
    def div_kernel(self, L):
        crL = tf.math.pow(L,1.0/3.0)
        self.u = tf.math.divide(self.u, crL)
        self.v = tf.math.divide(self.v, crL) 
        self.w = tf.math.divide(self.w, crL)

    def normalize_l2(self):
        self.div_kernel(l2_bound(self.construct_kern()))

    #Google paper inspired operator norm regularization
    def normalize_l2_op(self, clip_to):
        u_hat = tf.signal.fft(tf.cast(self.u, dtype = tf.complex64))
        u_hat_norm, _ = tf.linalg.normalize(u_hat, ord=2, axis=0)
        v_hat = tf.signal.fft(tf.cast(self.v, dtype = tf.complex64))
        v_hat_norm, _ = tf.linalg.normalize(v_hat, ord=2, axis=0)
        D, U, V = tf.linalg.svd(tf.cast(self.w, dtype = tf.complex64))
        D_clipped = tf.cast(tf.minimum(D, clip_to), tf.complex64)
        self.w = tf.math.real(tf.matmul(U, tf.matmul(tf.linalg.diag(D_clipped), V, adjoint_b=True)))
        self.u = tf.math.real(tf.signal.ifft(u_hat_norm))
        self.v = tf.math.real(tf.signal.ifft(v_hat_norm))
        
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
    def __init__(self, kern_size = 3, norm_flag=False, stride=1, **kwarg):
        if(len(kwarg)>2):
          super(SpaceSepConv2, self).__init__(name = kwarg['name'])
        else:
          super(SpaceSepConv2, self).__init__()
        self.u = self.add_weight(
            shape=([kwarg['out_channels'], kwarg['input_dim'][-1], kern_size]), initializer=tf.keras.initializers.glorot_normal, trainable=True
        )
        self.v = self.add_weight(
            shape=([kwarg['out_channels'], kwarg['input_dim'][-1], kern_size]), initializer=tf.keras.initializers.glorot_normal, trainable=True
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
        self.norm_flag = tf.Variable(
            initial_value=tf.constant(norm_flag), trainable=False
        )
        self.stride = tf.Variable(
            initial_value=tf.constant(stride), trainable=False
        )
        self.in_shape = tf.Variable(
            initial_value=tf.constant(kwarg['input_dim'][:-1]), trainable=False
        )

    def construct_kern(self):
        #Build kernel using u, v here
        #kern[i,j,k,l] = u[k,l,j]*w[k,l,i]*v[l,i]
        return tf.einsum('lkj,lki->ijkl', self.u, self.v)

    def call(self, inputs):    
        kern = self.construct_kern()
        return tf.keras.activations.relu(tf.nn.conv2d(inputs, kern, strides = self.stride, padding='SAME') + self.b)

    def div_kernel(self, L):
        srL = tf.math.pow(L,1.0/2.0)
        self.u = tf.math.divide(self.u, srL)
        self.v = tf.math.divide(self.v, srL) 

    #Normalize in the L_2 - L_2 lipschitz sense
    def normalize_l2(self):
        self.div_kernel(l2_bound(self.construct_kern()))

    def normalize_google(self, norm_bound):
        kern = self.construct_kern()
        D, U, V, _ =  SVD_Conv_Tensor(kern, self.in_shape.numpy())
        L = tf.math.reduce_max(D)
        self.div_kernel(L/norm_bound)

class DepthSepConv2(tf.keras.layers.Layer):
    def __init__(self, kern_size = 3, norm_flag=False, stride=1, **kwarg):
        if(len(kwarg)>2):
          super(DepthSepConv2, self).__init__(name = kwarg['name'])
        else:
          super(DepthSepConv2, self).__init__()
        self.k = self.add_weight(
            shape=([kwarg['out_channels'], kern_size, kern_size]), initializer=tf.keras.initializers.glorot_normal, trainable=True
        )
        self.u = self.add_weight(
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
        self.norm_flag = tf.Variable(
            initial_value=tf.constant(norm_flag), trainable=False
        )
        self.stride = tf.Variable(
            initial_value=tf.constant(stride), trainable=False
        )
        self.in_shape = (int(kwarg['input_dim'][1]), int(kwarg['input_dim'][2]))

    def construct_kern(self):
        #Build kernel using u, v here
        #kern[i,j,k,l] = u[k,l,j]*w[k,l,i]*v[l,i]
        return tf.einsum('lij,lk->ijkl', self.k, self.u)

    def call(self, inputs):    
        kern = self.construct_kern()
        return tf.keras.activations.relu(tf.nn.conv2d(inputs, kern, strides=self.stride, padding='SAME') + self.b)

    def div_kernel(self, L):
        srL = tf.math.pow(L,1.0/2.0)
        self.k = tf.math.divide(self.k, srL)
        self.u = tf.math.divide(self.u, srL) 

    #Normalize in the L_2 - L_2 lipschitz sense
    def normalize_l2(self):
        self.div_kernel(l2_bound(self.construct_kern()))

    def normalize_cust(self):
        self.div_kernel(self.cust_bound())

    def normalize_google(self):
        kern = self.construct_kern()
        D, U, V, _ =  full_svd(kern, self.in_shape)
        self.div_kernel(tf.math.reduce_max(D))

    def cust_bound(self):
        D = tf.linalg.svd(self.u, compute_uv=False)
        return tf.math.reduce_max(tf.norm(self.k, ord=2, axis=0))\
          *tf.math.reduce_max(D)\
          *tf.math.sqrt(tf.cast(self.out_channels, tf.float32))
        
