import numpy as np
import os, math, time, collections, numpy as np
''' TF_CPP_MIN_LOG_LEVEL
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
Disable Logs for now '''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import random as rn

# fix all randomness, except for multi-treading or GPU process
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
tf.set_random_seed(1234)

import tensorflow.contrib.slim as slim
import sys, shutil, subprocess

from lib.ops import *
from lib.dataloader import inference_data_loader, frvsr_gpu_data_loader
from lib.frvsr import generator_F, fnet
from lib.Teco import FRVSR, TecoGAN


Flags = tf.app.flags

Flags.DEFINE_integer('rand_seed', 1 , 'random seed' )

# Directories
Flags.DEFINE_string('input_dir_LR', None, 'The directory of the input resolution input data, for inference mode')
Flags.DEFINE_integer('input_dir_len', -1, 'length of the input for inference mode, -1 means all')
Flags.DEFINE_string('input_dir_HR', None, 'The directory of the input resolution input data, for inference mode')
Flags.DEFINE_string('mode', 'inference', 'train, or inference')
Flags.DEFINE_string('output_dir', None, 'The output directory of the checkpoint')
Flags.DEFINE_string('output_pre', '', 'The name of the subfolder for the images')
Flags.DEFINE_string('output_name', 'output', 'The pre name of the outputs')
Flags.DEFINE_string('output_ext', 'jpg', 'The format of the output when evaluating')
Flags.DEFINE_string('summary_dir', None, 'The dirctory to output the summary')

# Models
Flags.DEFINE_string('checkpoint', None, 'If provided, the weight will be restored from the provided checkpoint')
Flags.DEFINE_integer('num_resblock', 16, 'How many residual blocks are there in the generator')
# Models for training
Flags.DEFINE_boolean('pre_trained_model', False, 'If True, the weight of generator will be loaded as an initial point'
                                                     'If False, continue the training')
Flags.DEFINE_string('vgg_ckpt', None, 'path to checkpoint file for the vgg19')

# Machine resources
Flags.DEFINE_string('cudaID', '0', 'CUDA devices')
Flags.DEFINE_integer('queue_thread', 6, 'The threads of the queue (More threads can speedup the training process.')
Flags.DEFINE_integer('name_video_queue_capacity', 512, 'The capacity of the filename queue (suggest large to ensure'
                                                  'enough random shuffle.')
Flags.DEFINE_integer('video_queue_capacity', 256, 'The capacity of the video queue (suggest large to ensure'
                                                   'enough random shuffle')
Flags.DEFINE_integer('video_queue_batch', 2, 'shuffle_batch queue capacity')
                                                   
# Training details
# The data preparing operation
Flags.DEFINE_integer('RNN_N', 10, 'The number of the rnn recurrent length')
Flags.DEFINE_integer('batch_size', 4, 'Batch size of the input batch')
Flags.DEFINE_boolean('flip', True, 'Whether random flip data augmentation is applied')
Flags.DEFINE_boolean('random_crop', True, 'Whether perform the random crop')
Flags.DEFINE_boolean('movingFirstFrame', True, 'Whether use constant moving first frame randomly.')
Flags.DEFINE_integer('crop_size', 32, 'The crop size of the training image')
# Training data settings
Flags.DEFINE_string('input_video_dir', '', 'The directory of the video input data, for training')
Flags.DEFINE_string('input_video_pre', 'scene', 'The pre of the directory of the video input data')
Flags.DEFINE_integer('str_dir', 1000, 'The starting index of the video directory')
Flags.DEFINE_integer('end_dir', 2000, 'The ending index of the video directory')
Flags.DEFINE_integer('end_dir_val', 2050, 'The ending index for validation of the video directory')
Flags.DEFINE_integer('max_frm', 119, 'The ending index of the video directory')
# The loss parameters
Flags.DEFINE_float('vgg_scaling', -0.002, 'The scaling factor for the VGG perceptual loss, disable with negative value')
Flags.DEFINE_float('warp_scaling', 1.0, 'The scaling factor for the warp')
Flags.DEFINE_boolean('pingpang', False, 'use bi-directional recurrent or not')
Flags.DEFINE_float('pp_scaling', 1.0, 'factor of pingpang term, only works when pingpang is True')
# Training parameters
Flags.DEFINE_float('EPS', 1e-12, 'The eps added to prevent nan')
Flags.DEFINE_float('learning_rate', 0.0001, 'The learning rate for the network')
Flags.DEFINE_integer('decay_step', 500000, 'The steps needed to decay the learning rate')
Flags.DEFINE_float('decay_rate', 0.5, 'The decay rate of each decay step')
Flags.DEFINE_boolean('stair', False, 'Whether perform staircase decay. True => decay in discrete interval.')
Flags.DEFINE_float('beta', 0.9, 'The beta1 parameter for the Adam optimizer')
Flags.DEFINE_float('adameps', 1e-8, 'The eps parameter for the Adam optimizer')
Flags.DEFINE_integer('max_epoch', None, 'The max epoch for the training')
Flags.DEFINE_integer('max_iter', 1000000, 'The max iteration of the training')
Flags.DEFINE_integer('display_freq', 20, 'The diplay frequency of the training process')
Flags.DEFINE_integer('summary_freq', 100, 'The frequency of writing summary')
Flags.DEFINE_integer('save_freq', 10000, 'The frequency of saving images')
# Dst parameters
Flags.DEFINE_float('ratio', 0.01, 'The ratio between content loss and adversarial loss')
Flags.DEFINE_boolean('Dt_mergeDs', True, 'Whether only use a merged Discriminator.')
Flags.DEFINE_float('Dt_ratio_0', 1.0, 'The starting ratio for the temporal adversarial loss')
Flags.DEFINE_float('Dt_ratio_add', 0.0, 'The increasing ratio for the temporal adversarial loss')
Flags.DEFINE_float('Dt_ratio_max', 1.0, 'The max ratio for the temporal adversarial loss')
Flags.DEFINE_float('Dbalance', 0.4, 'An adaptive balancing for Discriminators')
Flags.DEFINE_float('crop_dt', 0.75, 'factor of dt crop') # dt input size = crop_size*crop_dt
Flags.DEFINE_boolean('D_LAYERLOSS', True, 'Whether use layer loss from D')

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
    variables_names = [ [v.name, v.get_shape().as_list()] for v in tf.get_collection(key, scope=scope)]
    total_sz = 0
    for k in variables_names:
        print ("Variable: " + k[0])
        print ("Shape: " + str(k[1]))
        total_sz += np.prod(k[1])
    print("total size: %d" %total_sz)
    
def preexec(): # Don't forward signals.
    os.setpgrp()
    
def testWhileTrain(FLAGS, testno = 0):
    '''
        this function is called during training, Hard-Coded!!
        to try the "inference" mode when a new model is saved.
        The code has to be updated from machine to machine...
        depending on python, and your training settings
    '''
    desstr = os.path.join(FLAGS.output_dir, 'train/') # saving in the ./train/ directory
    cmd1 = ["python3", "main.py", # never tested with python2...
        "--output_dir", desstr, 
        "--summary_dir", desstr,
        "--mode","inference",
        "--num_resblock", "%d"%FLAGS.num_resblock,
        "--checkpoint", os.path.join(FLAGS.output_dir, 'model-%d'%testno),
        "--cudaID", FLAGS.cudaID]
    # a folder for short test 
    cmd1 += ["--input_dir_LR", "./LR/calendar/", # update the testing sequence
             "--output_pre", "", # saving in train folder directly
             "--output_name", "%09d"%testno, # name
             "--input_dir_len", "10",]
    print('[testWhileTrain] step %d:'%testno)
    print(' '.join(cmd1))
    # ignore signals
    return subprocess.Popen(cmd1, preexec_fn = preexec)
    
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
    print("input shape:", input_shape)
    print("output shape:", output_shape)
    
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
    if (FLAGS.output_pre == ""):
        image_dir = FLAGS.output_dir
    else:
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
                filename = FLAGS.output_name+'_'+name
                print('saving image %s' % filename)
                out_path = os.path.join(image_dir, "%s.%s"%(filename,FLAGS.output_ext))
                save_img(out_path, output_frame[0])
            else:# First 5 is a hard-coded symmetric frame padding, ignored but time added!
                print("Warming up %d"%(5-i))
    print( "total time " + str(srtime) + ", frame number " + str(max_iter) )
        
# The training mode
elif FLAGS.mode == 'train':
    # hard coded save
    filelist = ['main.py','lib/Teco.py','lib/frvsr.py','lib/dataloader.py','lib/ops.py']
    for filename in filelist:
        shutil.copyfile('./' + filename, FLAGS.summary_dir + filename.replace("/","_"))
        
    useValidat = tf.placeholder_with_default( tf.constant(False, dtype=tf.bool), shape=() )
    rdata = frvsr_gpu_data_loader(FLAGS, useValidat)
    # Data = collections.namedtuple('Data', 'paths_HR, s_inputs, s_targets, image_count, steps_per_epoch')
    print('tData count = %d, steps per epoch %d' % (rdata.image_count, rdata.steps_per_epoch))
    if (FLAGS.ratio>0):
        Net = TecoGAN( rdata.s_inputs, rdata.s_targets, FLAGS )
    else:
        Net = FRVSR( rdata.s_inputs, rdata.s_targets, FLAGS )
    # Network = collections.namedtuple('Network', 'gen_output, train, learning_rate, update_list, '
    #                                     'update_list_name, update_list_avg, image_summary')
    
    # Add scalar summary
    tf.summary.scalar('learning_rate', Net.learning_rate)
    train_summary = []
    for key, value in zip(Net.update_list_name, Net.update_list_avg):
        # 'map_loss, scale_loss, FrameA_loss, FrameA_loss,...'
        train_summary += [tf.summary.scalar(key, value)]
    train_summary += Net.image_summary
    merged = tf.summary.merge(train_summary)
    
    validat_summary = [] # val data statistics is not added to average
    uplen = len(Net.update_list)
    for key, value in zip(Net.update_list_name[:uplen], Net.update_list):
        # 'map_loss, scale_loss, FrameA_loss, FrameA_loss,...'
        validat_summary += [tf.summary.scalar("val_" + key, value)]
    val_merged = tf.summary.merge(validat_summary)

    # Define the saver and weight initiallizer
    saver = tf.train.Saver(max_to_keep=50)
    # variable lists
    all_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    tfflag = tf.GraphKeys.MODEL_VARIABLES #tf.GraphKeys.TRAINABLE_VARIABLES
    
    if (FLAGS.checkpoint is not None) and (FLAGS.pre_trained_model is True):
        model_var_list = tf.get_collection(tfflag, scope='generator') + tf.get_collection(tfflag, scope='fnet')
        assign_ops = get_existing_from_ckpt(FLAGS.checkpoint, model_var_list, rest_zero=True, print_level=1)
        print('Prepare to load %d weights from the pre-trained model for generator and fnet'%len(assign_ops))
        if FLAGS.ratio>0:
            model_var_list = tf.get_collection(tfflag, scope='tdiscriminator')
            dis_list = get_existing_from_ckpt(FLAGS.checkpoint, model_var_list, print_level=0)
            print('Prepare to load %d weights from the pre-trained model for discriminator'%len(dis_list))
            assign_ops += dis_list
        
    if FLAGS.vgg_scaling > 0.0: # VGG weights are not trainable
        vgg_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg_19')
        vgg_restore = tf.train.Saver(vgg_var_list)
    
    print('Finish building the network.')
    
    # Start the session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # init_op = tf.initialize_all_variables() # MonitoredTrainingSession will initialize automatically
    with tf.train.MonitoredTrainingSession(config=config, save_summaries_secs=None, save_checkpoint_secs=None) as sess:
        train_writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)
        
        printVariable('generator')
        printVariable('fnet')
        if FLAGS.ratio>0:
            printVariable('tdiscriminator')
                
        if FLAGS.vgg_scaling > 0.0:
            printVariable('vgg_19', tf.GraphKeys.GLOBAL_VARIABLES)
            vgg_restore.restore(sess, FLAGS.vgg_ckpt)
            print('VGG19 restored successfully!!')
                
        if (FLAGS.checkpoint is not None):
            if (FLAGS.pre_trained_model is False):
                print('Loading everything from the checkpoint to continue the training...')
                saver.restore(sess, FLAGS.checkpoint)
                # this will restore everything, including ADAM training parameters and global_step
            else:
                print('Loading weights from the pre-trained model to start a new training...')
                sess.run(assign_ops) # only restore existing model weights
            
        print('The first run takes longer time for training data loading...')
        # get the session for save
        _sess = sess
        while type(_sess).__name__ != 'Session':
            # pylint: disable=W0212
            _sess = _sess._sess
        save_sess = _sess

        if 1:
            print('Save initial checkpoint, before any training')
            init_run_no = sess.run(Net.global_step)
            saver.save(save_sess, os.path.join(FLAGS.output_dir, 'model'), global_step=init_run_no)
            testWhileTrain(FLAGS, init_run_no) # make sure that testWhileTrain works
        
        # Performing the training
        frame_len = (FLAGS.RNN_N*2-1) if FLAGS.pingpang else FLAGS.RNN_N
        max_iter, step, start = FLAGS.max_iter, 0, time.time()
        if max_iter is None:
            if FLAGS.max_epoch is None:
                raise ValueError('one of max_epoch or max_iter should be provided')
            else:
                max_iter = FLAGS.max_epoch * rdata.steps_per_epoch
        try:
            for step in range(max_iter):
                run_step = sess.run(Net.global_step) + 1
                fetches = { "train": Net.train, "learning_rate": Net.learning_rate }
            
                if (run_step % FLAGS.display_freq) == 0:
                    for key, value in zip(Net.update_list_name, Net.update_list_avg):
                        fetches[str(key)] = value
                if (run_step % FLAGS.summary_freq) == 0:
                    fetches["summary"] = merged
                    
                results = sess.run(fetches)
                if(step == 0):
                    print('Optimization starts!!!(Ctrl+C to stop, will try saving the last model...)')
                
                if (run_step % FLAGS.summary_freq) == 0:
                    print('Run and Recording summary!!')
                    train_writer.add_summary(results['summary'], run_step)
                    val_fetches = {}
                    for name, value in zip(Net.update_list_name[:uplen], Net.update_list):
                        val_fetches['val_' + name] = value
                    val_fetches['summary'] = val_merged
                    val_results = sess.run(val_fetches, feed_dict={useValidat: True})
                    train_writer.add_summary(val_results['summary'], run_step)
                    print('-----------Validation data scalars-----------')
                    for name in Net.update_list_name[:uplen]:
                        print('val_' + name, val_results['val_' + name])
                        
                if (run_step % FLAGS.display_freq) == 0:
                    train_epoch = math.ceil(run_step / rdata.steps_per_epoch)
                    train_step = (run_step - 1) % rdata.steps_per_epoch + 1
                    rate = (step + 1) * FLAGS.batch_size / (time.time() - start)
                    remaining = (max_iter - step) * FLAGS.batch_size / rate
                    print("progress  epoch %d  step %d  image/sec %0.1fx%02d  remaining %dh%dm" % 
                        (train_epoch, train_step, rate, frame_len, 
                        remaining // 3600, (remaining%3600) // 60))
                        
                    print("global_step", run_step)
                    print("learning_rate", results['learning_rate'])
                    for name in Net.update_list_name:
                        print(name, results[name])
                        
                if (run_step % FLAGS.save_freq) == 0:
                    print('Save the checkpoint')
                    saver.save(save_sess, os.path.join(FLAGS.output_dir, 'model'), global_step=int(run_step))
                    testWhileTrain(FLAGS, run_step)
                
        except KeyboardInterrupt:
            if step > 1:
                print('main.py: KeyboardInterrupt->saving the checkpoint')
                saver.save(save_sess, os.path.join(FLAGS.output_dir, 'model'), global_step=int(run_step))
                testWhileTrain(FLAGS, run_step).communicate()
            print('main.py: quit')
            exit()
        print('Optimization done!!!!!!!!!!!!')
