import tensorflow as tf
import tensorflow.contrib.slim as slim
import pdb
import keras

import numpy as np, cv2 as cv, scipy
from scipy import signal
import collections
from tensorflow.python.ops import summary_op_util

### tensorflow functions ######################################################

def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2


def preprocessLR(image):
    with tf.name_scope("preprocessLR"):
        return tf.identity(image)


def deprocessLR(image):
    with tf.name_scope("deprocessLR"):
        return tf.identity(image)

# Define the convolution transpose building block
def conv2_tran(batch_input, kernel=3, output_channel=64, stride=1, use_bias=True, scope='conv'):
    # kernel: An integer specifying the width and height of the 2D convolution window
    with tf.variable_scope(scope):
        if use_bias:
            return slim.conv2d_transpose(batch_input, output_channel, [kernel, kernel], stride, 'SAME', data_format='NHWC',
                            activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())
        else:
            return slim.conv2d_transpose(batch_input, output_channel, [kernel, kernel], stride, 'SAME', data_format='NHWC',
                            activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(),
                            biases_initializer=None)

# Define the convolution building block
def conv2(batch_input, kernel=3, output_channel=64, stride=1, use_bias=True, scope='conv'):
    # kernel: An integer specifying the width and height of the 2D convolution window
    with tf.variable_scope(scope):
        if use_bias:
            return slim.conv2d(batch_input, output_channel, [kernel, kernel], stride, 'SAME', data_format='NHWC',
                            activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())
        else:
            return slim.conv2d(batch_input, output_channel, [kernel, kernel], stride, 'SAME', data_format='NHWC',
                            activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(),
                            biases_initializer=None)


def conv2_NCHW(batch_input, kernel=3, output_channel=64, stride=1, use_bias=True, scope='conv_NCHW'):
    # Use NCWH to speed up the inference
    # kernel: list of 2 integer specifying the width and height of the 2D convolution window
    with tf.variable_scope(scope):
        if use_bias:
            return slim.conv2d(batch_input, output_channel, [kernel, kernel], stride, 'SAME', data_format='NCWH',
                               activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())
        else:
            return slim.conv2d(batch_input, output_channel, [kernel, kernel], stride, 'SAME', data_format='NCWH',
                               activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(),
                               biases_initializer=None)


# Define our tensorflow version PRelu
def prelu_tf(inputs, name='Prelu'):
    with tf.variable_scope(name):
        alphas = tf.get_variable('alpha', inputs.get_shape()[-1], initializer=tf.zeros_initializer(), \
            collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.TRAINABLE_VARIABLES, tf.GraphKeys.MODEL_VARIABLES ],dtype=tf.float32)
    pos = tf.nn.relu(inputs)
    neg = alphas * (inputs - abs(inputs)) * 0.5

    return pos + neg


# Define our Lrelu
def lrelu(inputs, alpha):
    return keras.layers.LeakyReLU(alpha=alpha).call(inputs)


def batchnorm(inputs, is_training):
    return slim.batch_norm(inputs, decay=0.9, epsilon=0.001, updates_collections=tf.GraphKeys.UPDATE_OPS,
                        scale=False, fused=True, is_training=is_training)

def maxpool(inputs, scope='maxpool'):
    return slim.max_pool2d(inputs, [2, 2], scope=scope)
    
# Our dense layer
def denselayer(inputs, output_size):
    # Rachel todo, put it to Model variable_scope
    denseLayer = tf.layers.Dense(output_size, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
    output = denseLayer.apply(inputs)
    tf.add_to_collection( name=tf.GraphKeys.MODEL_VARIABLES, value=denseLayer.kernel )
    #output = tf.layers.dense(inputs, output_size, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
    
    return output

# The implementation of PixelShuffler
def pixelShuffler(inputs, scale=2):
    size = tf.shape(inputs)
    batch_size = size[0]
    h = size[1]
    w = size[2]
    c = inputs.get_shape().as_list()[-1]

    # Get the target channel size
    channel_target = c // (scale * scale)
    channel_factor = c // channel_target

    shape_1 = [batch_size, h, w, channel_factor // scale, channel_factor // scale]
    shape_2 = [batch_size, h * scale, w * scale, 1]

    # Reshape and transpose for periodic shuffling for each channel
    input_split = tf.split(inputs, channel_target, axis=3)
    output = tf.concat([phaseShift(x, scale, shape_1, shape_2) for x in input_split], axis=3)

    return output
    
def upscale_four(inputs, scope='upscale_four'): # mimic the tensorflow bilinear-upscaling for a fix ratio of 4
    with tf.variable_scope(scope):
        size = tf.shape(inputs)
        b = size[0]
        h = size[1]
        w = size[2]
        c = size[3]
        
        p_inputs = tf.concat( (inputs, inputs[:,-1:,:,:] ), axis = 1) # pad bottom
        p_inputs = tf.concat( (p_inputs, p_inputs[:,:,-1:,:] ), axis = 2) # pad right
        
        hi_res_bin = [ 
            [
                inputs, # top-left
                p_inputs[:,:-1,1:,:] # top-right
            ], 
            [
                p_inputs[:,1:,:-1,:], # bottom-left
                p_inputs[:,1:,1:,:] # bottom-right
            ]
        ]
        
        hi_res_array = [] 
        for hi in range(4):
            for wj in range(4):
                hi_res_array.append( 
                    hi_res_bin[0][0] * (1.0 - 0.25 * hi) * (1.0 - 0.25 * wj) 
                    + hi_res_bin[0][1] * (1.0 - 0.25 * hi) * (0.25 * wj) 
                    + hi_res_bin[1][0] * (0.25 * hi) * (1.0 - 0.25 * wj) 
                    + hi_res_bin[1][1] * (0.25 * hi) * (0.25 * wj) 
                    )
                    
        hi_res =  tf.stack( hi_res_array, axis = 3 ) # shape (b,h,w,16,c)
        hi_res_reshape = tf.reshape( hi_res, (b, h, w, 4, 4, c) )
        hi_res_reshape = tf.transpose( hi_res_reshape, (0,1,3,2,4,5) )
        hi_res_reshape = tf.reshape( hi_res_reshape, (b, h*4, w*4, c) )
    
    return hi_res_reshape
    
    
def bicubic_four(inputs, scope='bicubic_four'): 
    '''
        equivalent to tf.image.resize_bicubic( inputs, (h*4, w*4) ) for a fix ratio of 4 FOR API <=1.13
        For API 2.0, tf.image.resize_bicubic will be different, old version is tf.compat.v1.image.resize_bicubic
        **Parallel Catmull-Rom Spline Interpolation Algorithm for Image Zooming Based on CUDA*[Wu et. al.]**
    '''
    
    with tf.variable_scope(scope):
        size = tf.shape(inputs)
        b = size[0]
        h = size[1]
        w = size[2]
        c = size[3]
        
        p_inputs = tf.concat( (inputs[:,:1,:,:],   inputs)  , axis = 1) # pad top 
        p_inputs = tf.concat( (p_inputs[:,:,:1,:], p_inputs), axis = 2) # pad left
        p_inputs = tf.concat( (p_inputs, p_inputs[:,-1:,:,:], p_inputs[:,-1:,:,:]), axis = 1) # pad bottom
        p_inputs = tf.concat( (p_inputs, p_inputs[:,:,-1:,:], p_inputs[:,:,-1:,:]), axis = 2) # pad right
        
        hi_res_bin = [p_inputs[:,bi:bi+h,:,:] for bi in range(4) ]
        r = 0.75
        mat = np.float32( [[0,1,0,0],[-r,0,r,0], [2*r,r-3,3-2*r,-r], [-r,2-r,r-2,r]] )
        weights = [np.float32([1.0, t, t*t, t*t*t]).dot(mat) for t in [0.0, 0.25, 0.5, 0.75]]
        
        hi_res_array = [] # [hi_res_bin[1]] 
        for hi in range(4):
            cur_wei = weights[hi]
            cur_data = cur_wei[0] * hi_res_bin[0] + cur_wei[1] * hi_res_bin[1] + cur_wei[2] * hi_res_bin[2] + cur_wei[3] * hi_res_bin[3]
                
            hi_res_array.append(cur_data)
            
        hi_res_y =  tf.stack( hi_res_array, axis = 2 ) # shape (b,h,4,w,c)
        hi_res_y = tf.reshape( hi_res_y, (b, h*4, w+3, c) )
        
        hi_res_bin = [hi_res_y[:,:,bj:bj+w,:] for bj in range(4) ]
        
        hi_res_array = [] # [hi_res_bin[1]]
        for hj in range(4):
            cur_wei = weights[hj]
            cur_data = cur_wei[0] * hi_res_bin[0] + cur_wei[1] * hi_res_bin[1] + cur_wei[2] * hi_res_bin[2] + cur_wei[3] * hi_res_bin[3]
                
            hi_res_array.append(cur_data)
            
        hi_res =  tf.stack( hi_res_array, axis = 3 ) # shape (b,h*4,w,4,c)
        hi_res = tf.reshape( hi_res, (b, h*4, w*4, c) )
        
    return hi_res

def phaseShift(inputs, scale, shape_1, shape_2):
    # Tackle the condition when the batch is None
    X = tf.reshape(inputs, shape_1)
    X = tf.transpose(X, [0, 1, 3, 2, 4])

    return tf.reshape(X, shape_2)

# The random flip operation used for loading examples of one batch
def random_flip_batch(input, decision):
    f1 = tf.identity(input)
    f2 = tf.image.flip_left_right(input)
    output = tf.where(tf.less(decision, 0.5), f2, f1)

    return output

# The random flip operation used for loading examples
def random_flip(input, decision):
    f1 = tf.identity(input)
    f2 = tf.image.flip_left_right(input)
    output = tf.cond(tf.less(decision, 0.5), lambda: f2, lambda: f1)

    return output

# The operation used to print out the configuration
def print_configuration_op(FLAGS):
    print('[Configurations]:')
    for name, value in FLAGS.flag_values_dict().items():
        print('\t%s: %s'%(name, str(value)))
    print('End of configuration')
    

def copy_update_configuration(FLAGS, updateDict = {}):
    namelist = []
    valuelist = []
    for name, value in FLAGS.flag_values_dict().items():
        namelist += [name] 
        if( name in updateDict):
            valuelist += [updateDict[name]]
        else:
            valuelist += [value]
    Params = collections.namedtuple('Params', ",".join(namelist))
    tmpFLAGS = Params._make(valuelist)
    #print(tmpFLAGS)
    return tmpFLAGS
    
def compute_psnr(ref, target):
    ref = tf.cast(ref, tf.float32)
    target = tf.cast(target, tf.float32)
    diff = target - ref
    sqr = tf.multiply(diff, diff)
    err = tf.reduce_sum(sqr)
    v = tf.shape(diff)[0] * tf.shape(diff)[1] * tf.shape(diff)[2] * tf.shape(diff)[3]
    mse = err / tf.cast(v, tf.float32)
    psnr = 10. * (tf.log(255. * 255. / mse) / tf.log(10.))

    return psnr

# VGG19 component
def vgg_arg_scope(weight_decay=0.0005):
  """Defines the VGG arg scope.
  Args:
    weight_decay: The l2 regularization coefficient.
  Returns:
    An arg_scope.
  """
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=tf.zeros_initializer()):
    with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
        return arg_sc

# VGG19 net
def vgg_19(inputs,
           num_classes=1000,        # no effect
           is_training=False,       # no effect
           dropout_keep_prob=0.5,   # no effect
           spatial_squeeze=True,    # no effect
           scope='vgg_19',
           reuse = False,
           fc_conv_padding='VALID'):
  """Changed from the Oxford Net VGG 19-Layers version E Example.
  Note: Only offer features from conv1 until relu54, classification part is removed
  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output. Otherwise,
      the output prediction map will be (input / 32) - 6 in case of 'VALID' padding.
  Returns:
    the last op containing the log predictions and end_points dict.
  """
  with tf.variable_scope(scope, 'vgg_19', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 2, slim.conv2d, 64, 3, scope='conv1', reuse=reuse)
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, 3, scope='conv2',reuse=reuse)
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 4, slim.conv2d, 256, 3, scope='conv3', reuse=reuse)
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(net, 4, slim.conv2d, 512, 3, scope='conv4',reuse=reuse)
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      net = slim.repeat(net, 4, slim.conv2d, 512, 3, scope='conv5',reuse=reuse)
      net = slim.max_pool2d(net, [2, 2], scope='pool5')
      # fully_connected layers are skipped here! because we only need the feature maps
      #     from the previous layers
      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)

      return net, end_points
# vgg_19.default_image_size = 224


### Helper functions for data loading ############################################################
def gaussian_2dkernel(size=5, sig=1.):
    """
    Returns a 2D Gaussian kernel array with side length size and a sigma of sig
    """
    gkern1d = signal.gaussian(size, std=sig).reshape(size, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return (gkern2d/gkern2d.sum())
    
def tf_data_gaussDownby4( HRdata, sigma = 1.5 ):
    """
    tensorflow version of the 2D down-scaling by 4 with Gaussian blur
    sigma: the sigma used for Gaussian blur
    return: down-scaled data
    """
    k_w = 1 + 2 * int(sigma * 3.0)
    gau_k = gaussian_2dkernel(k_w, sigma)
    gau_0 = np.zeros_like(gau_k)
    gau_list = np.float32(  [
        [gau_k, gau_0, gau_0],
        [gau_0, gau_k, gau_0],
        [gau_0, gau_0, gau_k]]  ) # only works for RGB images!
    gau_wei = np.transpose( gau_list, [2,3,0,1] )
    
    with tf.device('/gpu:0'):
        fix_gkern = tf.constant( gau_wei, dtype = tf.float32, shape = [k_w, k_w, 3, 3], name='gauss_blurWeights' )
        # shape [batch_size, crop_h, crop_w, 3]
        cur_data = tf.nn.conv2d(HRdata, fix_gkern, strides=[1,4,4,1], padding="VALID", name='gauss_downsample_4')
    
        return cur_data
        
### Helper functions for model loading ############################################################        
def get_existing_from_ckpt(ckpt, var_list=None, rest_zero=False, print_level=1):
    reader = tf.train.load_checkpoint(ckpt)
    ops = []
    if(var_list is None):
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    for var in var_list:
        tensor_name = var.name.split(':')[0]
        if reader.has_tensor(tensor_name):
            npvariable = reader.get_tensor(tensor_name)
            if(print_level >= 2):
                print ("loading tensor: " + str(var.name) + ", shape " + str(npvariable.shape))
            if( var.shape != npvariable.shape ):
                raise ValueError('Wrong shape in for {} in ckpt,expected {}, got {}.'.format(var.name, str(var.shape),
                    str(npvariable.shape)))
            ops.append(var.assign(npvariable))
        else:
            if(print_level >= 1): print("variable not found in ckpt: " + var.name)
            if rest_zero:
                if(print_level >= 1): print("Assign Zero of " + str(var.shape))
                npzeros = np.zeros((var.shape))
                ops.append(var.assign(npzeros))
    return ops
    
# gif summary
"""gif_summary_v2.ipynb, Original file is located at
[a future version] https://colab.research.google.com/drive/1CSOrCK8-iQCZfs3CVchLE42C52M_3Sej
[current version]  https://colab.research.google.com/drive/1vgD2HML7Cea_z5c3kPBcsHUIxaEVDiIc
"""

def encode_gif(images, fps):
    """Encodes numpy images into gif string.
    Args:
      images: A 5-D `uint8` `np.array` (or a list of 4-D images) of shape
        `[batch_size, time, height, width, channels]` where `channels` is 1 or 3.
      fps: frames per second of the animation
    Returns:
      The encoded gif string.
    Raises:
      IOError: If the ffmpeg command returns an error.
    """
    from subprocess import Popen, PIPE
    h, w, c = images[0].shape
    cmd = ['ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-r', '%.02f' % fps,
        '-s', '%dx%d' % (w, h),
        '-pix_fmt', {1: 'gray', 3: 'rgb24'}[c],
        '-i', '-',
        '-filter_complex', '[0:v]split[x][z];[z]palettegen[y];[x][y]paletteuse',
        '-r', '%.02f' % fps,
        '-f', 'gif',
        '-']
    proc = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    for image in images:
        proc.stdin.write(image.tostring())
    out, err = proc.communicate()
    if proc.returncode:
        err = '\n'.join([' '.join(cmd), err.decode('utf8')])
        raise IOError(err)
    del proc
    return out


def py_gif_summary(tag, images, max_outputs, fps):
    """Outputs a `Summary` protocol buffer with gif animations.
    Args:
      tag: Name of the summary.
      images: A 5-D `uint8` `np.array` of shape `[batch_size, time, height, width,
        channels]` where `channels` is 1 or 3.
      max_outputs: Max number of batch elements to generate gifs for.
      fps: frames per second of the animation
    Returns:
      The serialized `Summary` protocol buffer.
    Raises:
      ValueError: If `images` is not a 5-D `uint8` array with 1 or 3 channels.
    """
    is_bytes = isinstance(tag, bytes)
    if is_bytes:
        tag = tag.decode("utf-8")
    images = np.asarray(images)
    if images.dtype != np.uint8:
        raise ValueError("Tensor must have dtype uint8 for gif summary.")
    if images.ndim != 5:
        raise ValueError("Tensor must be 5-D for gif summary.")
    batch_size, _, height, width, channels = images.shape
    if channels not in (1, 3):
        raise ValueError("Tensors must have 1 or 3 channels for gif summary.")
    
    summ = tf.Summary()
    num_outputs = min(batch_size, max_outputs)
    for i in range(num_outputs):
        image_summ = tf.Summary.Image()
        image_summ.height = height
        image_summ.width = width
        image_summ.colorspace = channels  # 1: grayscale, 3: RGB
        try:
            image_summ.encoded_image_string = encode_gif(images[i], fps)
        except (IOError, OSError) as e:
            tf.logging.warning("Unable to encode images to a gif string because either ffmpeg is "
                "not installed or ffmpeg returned an error: %s. Falling back to an "
                "image summary of the first frame in the sequence.", e)
            try:
                from PIL import Image  # pylint: disable=g-import-not-at-top
                import io  # pylint: disable=g-import-not-at-top
                with io.BytesIO() as output:
                    Image.fromarray(images[i][0]).save(output, "PNG")
                    image_summ.encoded_image_string = output.getvalue()
            except:
                tf.logging.warning("Gif summaries requires ffmpeg or PIL to be installed: %s", e)
                image_summ.encoded_image_string = "".encode('utf-8') if is_bytes else ""
        if num_outputs == 1:
            summ_tag = "{}/gif".format(tag)
        else:
            summ_tag = "{}/gif/{}".format(tag, i)
        summ.value.add(tag=summ_tag, image=image_summ)
    summ_str = summ.SerializeToString()
    return summ_str

def gif_summary(name, tensor, max_outputs, fps, collections=None, family=None):
    """Outputs a `Summary` protocol buffer with gif animations.
    Args:
      name: Name of the summary.
      tensor: A 5-D `uint8` `Tensor` of shape `[batch_size, time, height, width,
        channels]` where `channels` is 1 or 3.
      max_outputs: Max number of batch elements to generate gifs for.
      fps: frames per second of the animation
      collections: Optional list of tf.GraphKeys.  The collections to add the
        summary to.  Defaults to [tf.GraphKeys.SUMMARIES]
      family: Optional; if provided, used as the prefix of the summary tag name,
        which controls the tab name used for display on Tensorboard.
    Returns:
      A scalar `Tensor` of type `string`. The serialized `Summary` protocol
      buffer.
    """
    tensor = tf.image.convert_image_dtype(tensor, dtype=tf.uint8, saturate=True)
    # tensor = tf.convert_to_tensor(tensor)
    if summary_op_util.skip_summary():
        return tf.constant("")
    with summary_op_util.summary_scope(name, family, values=[tensor]) as (tag, scope):
          val = tf.py_func(
              py_gif_summary,
              [tag, tensor, max_outputs, fps],
              tf.string,
              stateful=False,
              name=scope)
          summary_op_util.collect(val, collections, [tf.GraphKeys.SUMMARIES])
    return val


### Numpy functions ##################################################################################
def save_img(out_path, img):
    img = np.clip(img*255.0, 0, 255).astype(np.uint8)
    cv.imwrite(out_path, img[:,:,::-1])
    
    
