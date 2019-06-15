import tensorflow as tf
from lib.ops import *

import cv2 as cv
import collections, os, math
import numpy as np
from scipy import signal

# The inference data loader. 
# should be a png sequence
def inference_data_loader(FLAGS):

    filedir = FLAGS.input_dir_LR
    downSP = False
    if (FLAGS.input_dir_LR is None) or (not os.path.exists(FLAGS.input_dir_LR)):
        if (FLAGS.input_dir_HR is None) or (not os.path.exists(FLAGS.input_dir_HR)):
            raise ValueError('Input directory not found')
        filedir = FLAGS.input_dir_HR
        downSP = True
        
    image_list_LR_temp = os.listdir(filedir)
    image_list_LR_temp = [_ for _ in image_list_LR_temp if _.endswith(".png")] 
    image_list_LR_temp = sorted(image_list_LR_temp) # first sort according to abc, then sort according to 123
    image_list_LR_temp.sort(key=lambda f: int(''.join(list(filter(str.isdigit, f))) or -1))
    if FLAGS.input_dir_len > 0:
        image_list_LR_temp = image_list_LR_temp[:FLAGS.input_dir_len]
        
    image_list_LR = [os.path.join(filedir, _) for _ in image_list_LR_temp]

    # Read in and preprocess the images
    def preprocess_test(name):
        im = cv.imread(name,3).astype(np.float32)[:,:,::-1]
        
        if downSP:
            icol_blur = cv.GaussianBlur( im, (0,0), sigmaX = 1.5)
            im = icol_blur[::4,::4,::]
        im = im / 255.0 #np.max(im)
        return im

    image_LR = [preprocess_test(_) for _ in image_list_LR]
    
    if True: # a hard-coded symmetric padding
        image_list_LR = image_list_LR[5:0:-1] + image_list_LR
        image_LR = image_LR[5:0:-1] + image_LR

    Data = collections.namedtuple('Data', 'paths_LR, inputs')
    return Data(
        paths_LR=image_list_LR,
        inputs=image_LR
    )

# load hi-res data from disk
def loadHR_batch(FLAGS, tar_size):
    # file processing on CPU
    with tf.device('/cpu:0'):
        #Check the input directory
        if (FLAGS.input_video_dir == ''):
            raise ValueError('Video input directory input_video_dir is not provided')
        if (not os.path.exists(FLAGS.input_video_dir)):
            raise ValueError('Video input directory not found')
            
        image_set_lists = []
        with tf.variable_scope('load_frame'):
            for dir_i in range(FLAGS.str_dir, FLAGS.end_dir+1):
                inputDir = os.path.join( FLAGS.input_video_dir, '%s_%04d' %(FLAGS.input_video_pre, dir_i) )
                if (os.path.exists(inputDir)): # the following names are hard coded: col_high_
                    if not os.path.exists(os.path.join(inputDir ,'col_high_%04d.png' % FLAGS.max_frm) ):
                        print("Skip %s, since foler doesn't contain enough frames!" % inputDir)
                        continue
                        
                    image_list = [ os.path.join(inputDir ,'col_high_%04d.png' % frame_i ) 
                                    for frame_i in range(FLAGS.max_frm + 1 )]
                    image_set_lists.append(image_list) 
            tensor_set_lists = tf.convert_to_tensor(image_set_lists, dtype=tf.string)
            input_slices = tf.train.slice_input_producer([tensor_set_lists], shuffle=False,
                                            capacity=int(FLAGS.name_video_queue_capacity) )
            
            input_slices = input_slices[0] # one slice contains pathes for (FLAGS.max_frm frames + 1) frames
            
            HR_data = []
            for frame_i in range(FLAGS.max_frm + 1 ):
                HR_data_i =  tf.image.convert_image_dtype( 
                                    tf.image.decode_png(tf.read_file(input_slices[frame_i]), channels=3), 
                                    dtype=tf.float32)
                HR_data += [HR_data_i]
            # the reshape after loading is necessary as the unkown output shape crashes the cropping op
            HR_data = tf.stack(HR_data, axis=0)
            HR_shape = tf.shape(HR_data)#(FLAGS.max_frm+1, h, w, 3))
            HR_data = tf.reshape(HR_data, (HR_shape[0], HR_shape[1], HR_shape[2], HR_shape[3]))
            
        # sequence preparation and data augmentation part
        with tf.name_scope('sequence_data_preprocessing'):
            sequence_length = FLAGS.max_frm - FLAGS.RNN_N + 1
            num_image_list_HR_t_cur = len(image_set_lists)*sequence_length
            HR_sequences, name_sequences = [], []
            if FLAGS.random_crop and FLAGS.mode == 'train':
                print('[Config] Use random crop')
                # a k_w_border margin is in tar_size for gaussian blur
                # have to use the same crop because crop_to_bounding_box only accept one value
                offset_w = tf.cast(tf.floor(tf.random_uniform([], 0, \
                    tf.cast(HR_shape[-2], tf.float32) - tar_size )),dtype=tf.int32)
                offset_h = tf.cast(tf.floor(tf.random_uniform([], 0, \
                    tf.cast(HR_shape[-3], tf.float32) - tar_size )),dtype=tf.int32) 
            else:
                raise Exception('Not implemented') # train_data can have different resolutions, crop is necessary    
            
            if(FLAGS.movingFirstFrame): 
                print('[Config] Move first frame')
                # our data augmentation, moving first frame to mimic camera motion
                # random motions, one slice use the same motion
                offset_xy = tf.cast(tf.floor(tf.random_uniform([FLAGS.RNN_N, 2], -3.5,4.5)),dtype=tf.int32)
                # [FLAGS.RNN_N , 2], relative positions
                pos_xy = tf.cumsum(offset_xy, axis=0, exclusive=True) 
                # valid regions, lefttop_pos, target_size-range_pos
                min_pos = tf.reduce_min( pos_xy, axis=0 )
                range_pos = tf.reduce_max( pos_xy, axis=0 ) - min_pos # [ shrink x, shrink y ]
                lefttop_pos = pos_xy - min_pos # crop point
                moving_decision = tf.random_uniform([sequence_length], 0, 1, dtype=tf.float32)
                fix_off_h = tf.clip_by_value(offset_h, 0, HR_shape[-3] - tar_size - range_pos[1])
                fix_off_w = tf.clip_by_value(offset_w, 0, HR_shape[-2] - tar_size - range_pos[0])
            
            if FLAGS.flip and (FLAGS.mode == 'train'):
                print('[Config] Use random flip')
                # Produce the decision of random flip
                flip_decision = tf.random_uniform([sequence_length], 0, 1, dtype=tf.float32)
                
            for fi in range( FLAGS.RNN_N ):
                HR_sequence = HR_data[ fi : fi+sequence_length ]  # sequence_length, h, w, 3
                name_sequence = input_slices[ fi : fi+sequence_length ]
                
                if (FLAGS.flip is True) and (FLAGS.mode == 'train'):
                    HR_sequence = random_flip_batch(HR_sequence, flip_decision)
                    
                # currently, it is necessary to crop, because training videos have different resolutions
                if FLAGS.random_crop and FLAGS.mode == 'train':
                    HR_sequence_crop = tf.image.crop_to_bounding_box(HR_sequence, 
                            offset_h, offset_w, tar_size, tar_size)
                            
                    if FLAGS.movingFirstFrame:
                        if(fi == 0): # always use the first frame
                            HR_data_0 = tf.identity(HR_sequence)
                            name_0 = tf.identity(name_sequence)
                            
                        name_sequence1 = name_0
                        HR_sequence_crop_1 = tf.image.crop_to_bounding_box(HR_data_0, 
                            fix_off_h + lefttop_pos[fi][1], fix_off_w + lefttop_pos[fi][0], tar_size, tar_size)
                            
                        HR_sequence_crop = tf.where(moving_decision < 0.7, HR_sequence_crop, HR_sequence_crop_1)
                        name_sequence = tf.where(moving_decision < 0.7, name_sequence, name_sequence1) 
                

                HR_sequence_crop.set_shape([sequence_length, tar_size, tar_size, 3])
                HR_sequences.append( HR_sequence_crop )
                name_sequences.append( name_sequence )
                
            target_images = HR_sequences # RNN_N, sequence_length,tar_size,tar_size,3
            output_names = name_sequences
            
        if len(target_images)!=FLAGS.RNN_N:
            raise ValueError('Length of target image sequence is incorrect,expected {}, got {}.'.format(FLAGS.RNN_N, len(target_images)))
        
        print('Sequenced batches: {}, sequence length: {}'.format(num_image_list_HR_t_cur, FLAGS.RNN_N))
        batch_list = tf.train.shuffle_batch(output_names + target_images, enqueue_many=True,\
                        batch_size=int(FLAGS.batch_size), capacity=FLAGS.video_queue_capacity+FLAGS.video_queue_batch*sequence_length,
                        min_after_dequeue=FLAGS.video_queue_capacity, num_threads=FLAGS.queue_thread, seed = FLAGS.rand_seed)
        
    return batch_list, num_image_list_HR_t_cur # a k_w_border margin is in there for gaussian blur
    
# load hi-res data from disk
def loadHR(FLAGS, tar_size):
    # a k_w_border margin should be in tar_size for Gaussian blur
    
    with tf.device('/cpu:0'):
        #Check the input directory
        if (FLAGS.input_video_dir == ''):
            raise ValueError('Video input directory input_video_dir is not provided')

        if (not os.path.exists(FLAGS.input_video_dir)):
            raise ValueError('Video input directory not found')

        image_list_HR_r = [[] for _ in range( FLAGS.RNN_N )] # all empty lists
        
        for dir_i in range(FLAGS.str_dir, FLAGS.end_dir+1):
            inputDir = os.path.join( FLAGS.input_video_dir, '%s_%04d' %(FLAGS.input_video_pre, dir_i) )
            if (os.path.exists(inputDir)): # the following names are hard coded
                if not os.path.exists(os.path.join(inputDir ,'col_high_%04d.png' % FLAGS.max_frm) ):
                    print("Skip %s, since foler doesn't contain enough frames!" % inputDir)
                    continue
                for fi in range( FLAGS.RNN_N ):
                    image_list_HR_r[fi] += [ os.path.join(inputDir ,'col_high_%04d.png' % frame_i ) 
                                        for frame_i in range(fi, FLAGS.max_frm - FLAGS.RNN_N + fi + 1 )]
        
        num_image_list_HR_t_cur = len(image_list_HR_r[0])
        if num_image_list_HR_t_cur==0:
            raise Exception('No frame files in the video input directory')
                    
        image_list_HR_r = [tf.convert_to_tensor(_ , dtype=tf.string) for _ in image_list_HR_r ]

        with tf.variable_scope('load_frame'):
            # define the image list queue
            output = tf.train.slice_input_producer(image_list_HR_r, shuffle=False,\
                capacity=int(FLAGS.name_video_queue_capacity) )
            output_names = output
            
            data_list_HR_r = []# high res rgb, in range 0-1, shape any
                
            if FLAGS.movingFirstFrame and FLAGS.mode == 'train': # our data augmentation, moving first frame to mimic camera motion
                print('[Config] Use random crop')
                offset_xy = tf.cast(tf.floor(tf.random_uniform([FLAGS.RNN_N, 2], -3.5,4.5)),dtype=tf.int32)
                # [FLAGS.RNN_N , 2], shifts
                pos_xy = tf.cumsum(offset_xy, axis=0, exclusive=True) # relative positions
                min_pos = tf.reduce_min( pos_xy, axis=0 )
                range_pos = tf.reduce_max( pos_xy, axis=0 ) - min_pos # [ shrink x, shrink y ]
                lefttop_pos = pos_xy - min_pos # crop point
                moving_decision = tf.random_uniform([], 0, 1, dtype=tf.float32)
                
            for fi in range( FLAGS.RNN_N ):
                HR_data = tf.image.convert_image_dtype( tf.image.decode_png(tf.read_file(output[fi]), channels=3), dtype=tf.float32)
                if(FLAGS.movingFirstFrame):
                    if(fi == 0):
                        HR_data_0 = tf.identity(HR_data)
                        target_size = tf.shape(HR_data_0)
                        
                    HR_data_1 = tf.image.crop_to_bounding_box(HR_data_0, 
                        lefttop_pos[fi][1], lefttop_pos[fi][0], 
                        target_size[0] - range_pos[1], target_size[1] - range_pos[0])
                    HR_data = tf.cond(moving_decision < 0.7, lambda: tf.identity(HR_data), lambda: tf.identity(HR_data_1))
                    output_names[fi] = tf.cond(moving_decision < 0.7, lambda: tf.identity(output_names[fi]), lambda: tf.identity(output_names[0]))
                data_list_HR_r.append( HR_data )
            
            target_images = data_list_HR_r
        
        # Other data augmentation part
        with tf.name_scope('data_preprocessing'):
            
            with tf.name_scope('random_crop'):
                # Check whether perform crop
                if (FLAGS.random_crop is True) and FLAGS.mode == 'train':
                    print('[Config] Use random crop')
                    target_size = tf.shape(target_images[0])
                    
                    offset_w = tf.cast(tf.floor(tf.random_uniform([], 0, \
                        tf.cast(target_size[1], tf.float32) - tar_size )),dtype=tf.int32)
                    offset_h = tf.cast(tf.floor(tf.random_uniform([], 0, \
                        tf.cast(target_size[0], tf.float32) - tar_size )),dtype=tf.int32) 
                    
                    for frame_t in range(FLAGS.RNN_N):
                        target_images[frame_t] = tf.image.crop_to_bounding_box(target_images[frame_t], 
                            offset_h, offset_w, tar_size, tar_size) 
                        
                else:
                    raise Exception('Not implemented')
            
            with tf.variable_scope('random_flip'):
                # Check for random flip:
                if (FLAGS.flip is True) and (FLAGS.mode == 'train'):
                    print('[Config] Use random flip')
                    # Produce the decision of random flip
                    flip_decision = tf.random_uniform([], 0, 1, dtype=tf.float32)
                    for frame_t in range(FLAGS.RNN_N):
                        target_images[frame_t] = random_flip(target_images[frame_t], flip_decision)
                    
            for frame_t in range(FLAGS.RNN_N):
                target_images[frame_t].set_shape([tar_size, tar_size, 3])
                
        if FLAGS.mode == 'train':
            print('Sequenced batches: {}, sequence length: {}'.format(num_image_list_HR_t_cur, FLAGS.RNN_N))
            batch_list = tf.train.shuffle_batch(output_names + target_images,\
                            batch_size=int(FLAGS.batch_size), capacity=FLAGS.video_queue_capacity+FLAGS.video_queue_batch*FLAGS.max_frm,
                            min_after_dequeue=FLAGS.video_queue_capacity, num_threads=FLAGS.queue_thread, seed = FLAGS.rand_seed)
        else:
            raise Exception('Not implemented')
    return batch_list, num_image_list_HR_t_cur # a k_w_border margin is still there for gaussian blur!!


def frvsr_gpu_data_loader(FLAGS, useValData_ph): # useValData_ph, tf bool placeholder, whether to use validationdata
    Data = collections.namedtuple('Data', 'paths_HR, s_inputs, s_targets, image_count, steps_per_epoch')
    tar_size = FLAGS.crop_size
    tar_size = (FLAGS.crop_size * 4 ) + int(1.5 * 3.0) * 2 # crop_size * 4, and Gaussian blur margin
    k_w_border = int(1.5 * 3.0)
    
    loadHRfunc = loadHR if FLAGS.queue_thread > 4 else loadHR_batch
    # loadHR_batch load 120 frames at once, is faster for a single queue, and usually will be slower and slower for larger queue_thread
    # loadHR load RNN_N frames at once, is faster when queue_thread > 4, but slow for queue_thread < 4
    
    with tf.name_scope('load_frame_cpu'):
        with tf.name_scope('train_data'):
            print("Preparing train_data")
            batch_list, num_image_list_HR_t_cur = loadHRfunc(FLAGS, tar_size)
        with tf.name_scope('validation_data'):
            print("Preparing validation_data")
            val_capacity = 128 # TODO parameter!
            val_q_thread = 1   # TODO parameter!
            valFLAGS = copy_update_configuration(FLAGS, \
                {"str_dir":FLAGS.end_dir + 1,"end_dir":FLAGS.end_dir_val,"name_video_queue_capacity":val_capacity,\
                    "video_queue_capacity":val_capacity, "queue_thread":val_q_thread})
            vald_batch_list, vald_num_image_list_HR_t_cur = loadHRfunc(valFLAGS, tar_size)
            
    HR_images = list(batch_list[FLAGS.RNN_N::])# batch high-res images
    HR_images_vald = list(vald_batch_list[FLAGS.RNN_N::])# test batch high-res images
    
    steps_per_epoch = num_image_list_HR_t_cur // FLAGS.batch_size
    
    target_images = []
    input_images = []
    with tf.name_scope('load_frame_gpu'):
        with tf.device('/gpu:0'):
            for frame_t in range(FLAGS.RNN_N):
                def getTrainHR():
                    return HR_images[frame_t]
                def getValdHR():
                    return HR_images_vald[frame_t]
                    
                curHR = tf.cond( useValData_ph, getValdHR, getTrainHR )
                input_images.append( tf_data_gaussDownby4(curHR, 1.5) )
                
                input_images[frame_t].set_shape([FLAGS.batch_size,FLAGS.crop_size, FLAGS.crop_size, 3])
                input_images[frame_t] = preprocessLR(input_images[frame_t])
                
                target_images.append(tf.image.crop_to_bounding_box(curHR, 
                                k_w_border, k_w_border, \
                                FLAGS.crop_size*4,\
                                FLAGS.crop_size*4) )
                target_images[frame_t] = preprocess(target_images[frame_t])
                target_images[frame_t].set_shape([FLAGS.batch_size,FLAGS.crop_size*4, FLAGS.crop_size*4, 3])
            
            
            # for Ds, inputs_batch and targets_batch are just the input and output:
            S_inputs_frames = tf.stack(input_images, axis = 1) # batch, frame, FLAGS.crop_size, FLAGS.crop_size, sn
            S_targets_frames = tf.stack(target_images, axis = 1) # batch, frame, FLAGS.crop_size*4, FLAGS.crop_size*4, 3
            S_inputs_frames.set_shape( (FLAGS.batch_size,FLAGS.RNN_N,FLAGS.crop_size,FLAGS.crop_size,3) )
            S_targets_frames.set_shape( (FLAGS.batch_size,FLAGS.RNN_N,4*FLAGS.crop_size,4*FLAGS.crop_size,3) )
        
    #Data = collections.namedtuple('Data', 'paths_HR, s_inputs, s_targets, image_count, steps_per_epoch')
    def getTrainHRpath():
        return batch_list[:FLAGS.RNN_N]
    def getValdHRpath():
        return vald_batch_list[:FLAGS.RNN_N]
        
    curHRpath = tf.cond( useValData_ph, getValdHRpath, getTrainHRpath )
                
    return Data(
        paths_HR=curHRpath,
        s_inputs=S_inputs_frames,       # batch, frame, FLAGS.crop_size, FLAGS.crop_size, sn
        s_targets=S_targets_frames,     # batch, frame, FLAGS.crop_size*4, FLAGS.crop_size*4, 3
        image_count=num_image_list_HR_t_cur,
        steps_per_epoch=steps_per_epoch
    )
    
