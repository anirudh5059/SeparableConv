import tensorflow as tf


#Helper function to convert between the 5 channel format that conv3D uses and thr required 4 channel format we need
def process_3d(inputs, kern, b):
  out = tf.keras.activations.relu(tf.nn.conv3d(tf.transpose(inputs[..., None], perm = [0,3,1,2,4]), kern, [1,1,1,1,1], padding='SAME') + b)
  out_hat = tf.transpose(out, perm = [0,2,3,1,4])
  shape_oh = out_hat.get_shape().as_list()
  return tf.reshape(out_hat, [-1, shape_oh[1], shape_oh[2], shape_oh[3]*shape_oh[4]])


class Conv3(tf.keras.layers.Layer):
  def __init__(self, out_mult = 1, kern_size = 5, kern_depth = 3, **kwarg):
    if(len(kwarg)>1):
          super(Conv3, self).__init__(name = kwarg['name'])
    else:
          super(Conv3, self).__init__()
    self.u = self.add_weight(
      shape=([out_mult, kern_size]), initializer=tf.keras.initializers.glorot_normal, trainable=True
    )
    self.v = self.add_weight(
      shape=([out_mult, kern_size]), initializer=tf.keras.initializers.glorot_normal, trainable=True
    )
    self.w = self.add_weight(
      shape=([out_mult, kern_depth]), initializer=tf.keras.initializers.glorot_normal, trainable=True
    )
    self.in_channels = tf.Variable(
      initial_value=tf.constant(kwarg['input_dim'][-1]), trainable=False
    )
    self.out_mult = tf.Variable(
      initial_value=tf.constant(out_mult ), trainable=False
    )
    self.out_channels = tf.Variable(
      initial_value=tf.constant(out_mult*kwarg['input_dim'][-1]), trainable=False
    )
    self.kern_size = tf.Variable(
      initial_value = tf.constant(kern_size), trainable=False
    )
    self.kern_depth = tf.Variable(
        initial_value = tf.constant(kern_depth), trainable=False
    )
    self.b = self.add_weight(
      shape=([kwarg['input_dim'][-1]]+kwarg['input_dim'][-3:-1]+[out_mult]), initializer=tf.keras.initializers.zeros, trainable=True
    )

  def call(self, inputs):
    #Build kernel using u, v, w here
    #kern[k,i,j,0,l] = u[l,j]*w[l,k]*v[l,i]
    kern = tf.einsum('lj,lk,li,m->kijml', self.u, self.w, self.v, tf.constant(1, shape = [1,], dtype=tf.float32))
    
    #convert the convolution output from 5 dimensions to 4 while preserving result
    return process_3d(inputs, kern, self.b)

class FullConv3(tf.keras.layers.Layer):
  def __init__(self, out_mult = 1, kern_size = 5, kern_depth = 3, **kwarg):
    if(len(kwarg)>1):
          super(FullConv3, self).__init__(name = kwarg['name'])
    else:
          super(FullConv3, self).__init__()
    self.k = self.add_weight(
      shape=([out_mult, kern_depth, kern_size, kern_size]), initializer=tf.keras.initializers.glorot_normal, trainable=True
    )
    self.in_channels = tf.Variable(
      initial_value=tf.constant(kwarg['input_dim'][-1]), trainable=False
    )
    self.out_mult = tf.Variable(
      initial_value=tf.constant(out_mult ), trainable=False
    )
    self.out_channels = tf.Variable(
      initial_value=tf.constant(out_mult*kwarg['input_dim'][-1]), trainable=False
    )
    self.kern_size = tf.Variable(
      initial_value = tf.constant(kern_size), trainable=False
    )
    self.kern_depth = tf.Variable(
        initial_value = tf.constant(kern_depth), trainable=False
    )
    self.b = self.add_weight(
      shape=([kwarg['input_dim'][-1]]+kwarg['input_dim'][-3:-1]+[out_mult]), initializer=tf.keras.initializers.zeros, trainable=True
    )

  def call(self, inputs):
    #Build kernel using k here
    #kern[k,i,j,0,l] = k[l,k,i,j]

    kern = tf.einsum('lkij,m->kijml', self.k, tf.constant(1, shape = [1,], dtype=tf.float32))

    ##convert the convolution output from 5 dimensions to 4 while preserving result
    return process_3d(inputs, kern, self.b)