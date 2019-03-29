import numpy as np
import tensorflow as tf
import random as rn

import os, math, time, collections, numpy as np

# fix all randomness, except for multi-treading or GPU process
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
tf.set_random_seed(1234)

import tensorflow.contrib.slim as slim
import sys, shutil, subprocess

from lib.ops import *
from lib.dataloader import inference_data_loader
from lib.frvsr import generator_F, fnet


Flags = tf.app.flags

Flags.DEFINE_integer('rand_seed', 1 , 'random seed' )

# Directories
Flags.DEFINE_string('input_dir_LR', None, 'The directory of the input resolution input data, for inference mode')
Flags.DEFINE_string('input_dir_HR', None, 'The directory of the input resolution input data, for inference mode')
Flags.DEFINE_string('mode', 'inference', 'train, or inference')
Flags.DEFINE_string('output_dir', None, 'The output directory of the checkpoint')
Flags.DEFINE_string('output_pre', 'images', 'The output pre of the images')
Flags.DEFINE_string('output_ext', 'jpg', 'The format of the output when evaluating')
Flags.DEFINE_string('summary_dir', None, 'The dirctory to output the summary')

# Models
Flags.DEFINE_string('checkpoint', None, 'If provided, the weight will be restored from the provided checkpoint')
Flags.DEFINE_integer('num_resblock', 16, 'How many residual blocks are there in the generator')

# Machine resources
Flags.DEFINE_string('cudaID', '0', 'CUDA devices')

FLAGS = Flags.FLAGS

# Set CUDA devices correctly if you use multiple gpu system
os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.cudaID 
# Fix randomness
my_seed = FLAGS.rand_seed
rn.seed(my_seed)
np.random.seed(my_seed)
tf.set_random_seed(my_seed)

# Check the output_dir is given
if FLAGS.output_dir is None:
    raise ValueError('The output directory is needed')
# Check the output directory to save the checkpoint
if not os.path.exists(FLAGS.output_dir):
    os.mkdir(FLAGS.output_dir)
# Check the summary directory to save the event
if not os.path.exists(FLAGS.summary_dir):
    os.mkdir(FLAGS.summary_dir)

# custom Logger to write Log to file
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(FLAGS.summary_dir + "logfile.txt", "a") 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message) 
    def flush(self):
        self.log.flush()
        
sys.stdout = Logger()

def printVariable(scope, key = tf.GraphKeys.MODEL_VARIABLES):
    print("Scope %s:" % scope)
    variables_names = [v.name for v in tf.get_collection(key, scope=scope)]
    #variables_names = sorted(variables_names)
    values = sess.run(variables_names)
    total_sz = 0
    for k, v in zip(variables_names, values):
        print ("Variable: " + k)
        print ("Shape: " + str(v.shape))
        total_sz += np.prod(v.shape)
    print("total size: %d" %total_sz)
    
if False: # If you want to take a look of the configuration, True
    print_configuration_op(FLAGS)

# the inference mode (just perform super resolution on the input image)
if FLAGS.mode == 'inference':
    if FLAGS.checkpoint is None:
        raise ValueError('The checkpoint file is needed to performing the test.')

    # Declare the test data reader
    inference_data = inference_data_loader(FLAGS)
    input_shape = [1,] + list(inference_data.inputs[0].shape)
    output_shape = [1,input_shape[1]*4, input_shape[2]*4, 3]
    oh = input_shape[1] - input_shape[1]//8 * 8
    ow = input_shape[2] - input_shape[2]//8 * 8
    paddings = tf.constant([[0,0], [0,oh], [0,ow], [0,0]])
    print(input_shape)
    print(output_shape)
    # build the graph
    inputs_raw = tf.placeholder(tf.float32, shape=input_shape, name='inputs_raw')
    
    pre_inputs = tf.Variable(tf.zeros(input_shape), trainable=False, name='pre_inputs')
    pre_gen = tf.Variable(tf.zeros(output_shape), trainable=False, name='pre_gen')
    pre_warp = tf.Variable(tf.zeros(output_shape), trainable=False, name='pre_warp')
    
    transpose_pre = tf.space_to_depth(pre_warp, 4)
    inputs_all = tf.concat( (inputs_raw, transpose_pre), axis = -1)
    with tf.variable_scope('generator'):
        gen_output = generator_F(inputs_all, 3, reuse=False, FLAGS=FLAGS)
        # Deprocess the images outputed from the model, and assign things for next frame
        with tf.control_dependencies([ tf.assign(pre_inputs, inputs_raw)]):
            outputs = tf.assign(pre_gen, deprocess(gen_output))
    
    inputs_frames = tf.concat( (pre_inputs, inputs_raw), axis = -1)
    with tf.variable_scope('fnet'):
        gen_flow_lr = fnet( inputs_frames, reuse=False)
        gen_flow_lr = tf.pad(gen_flow_lr, paddings, "SYMMETRIC") 
        gen_flow = upscale_four(gen_flow_lr*4.0)
        gen_flow.set_shape( output_shape[:-1]+[2] )
    pre_warp_hi = tf.contrib.image.dense_image_warp(pre_gen, gen_flow)
    before_ops = tf.assign(pre_warp, pre_warp_hi)

    print('Finish building the network')
    
    # In inference time, we only need to restore the weight of the generator
    var_list = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope='generator')
    var_list = var_list + tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope='fnet')
    
    weight_initiallizer = tf.train.Saver(var_list)
    
    # Define the initialization operation
    init_op = tf.global_variables_initializer()
    local_init_op = tf.local_variables_initializer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    image_dir = os.path.join(FLAGS.output_dir, FLAGS.output_pre)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
        
    with tf.Session(config=config) as sess:
        # Load the pretrained model
        sess.run(init_op)
        sess.run(local_init_op)
        
        print('Loading weights from ckpt model')
        weight_initiallizer.restore(sess, FLAGS.checkpoint)
        if False: # If you want to take a look of the weights, True
            printVariable('generator')
            printVariable('fnet')
        max_iter = len(inference_data.inputs)
        
        srtime = 0
        print('Frame evaluation starts!!')
        for i in range(max_iter):
            input_im = np.array([inference_data.inputs[i]]).astype(np.float32)
            feed_dict={inputs_raw: input_im}
            t0 = time.time()
            if(i != 0):
                sess.run(before_ops, feed_dict=feed_dict)
            output_frame = sess.run(outputs, feed_dict=feed_dict)
            srtime += time.time()-t0
            
            if(i >= 5): 
                name, _ = os.path.splitext(os.path.basename(str(inference_data.paths_LR[i])))
                filename = "output_"+name
                print('saving image %s' % filename)
                out_path = os.path.join(image_dir, "%s.%s"%(filename,FLAGS.output_ext))
                save_img(out_path, output_frame[0])
            else:# First 5 is a hard-coded symmetric frame padding, ignored but time added!
                print("Warming up %d"%(5-i))
    print( "total time " + str(srtime) + ", frame number " + str(max_iter) )
        
# The training mode
elif FLAGS.mode == 'train':
    print('Comming Soon!')








