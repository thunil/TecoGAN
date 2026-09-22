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

Flags.DEFINE_integer('rand_seed', 1 , '随机种子' )

# Directories
Flags.DEFINE_string('input_dir_LR', None, '推理模式下输入低分辨率数据目录')
Flags.DEFINE_integer('input_dir_len', -1, '推理模式下输入序列长度，-1 表示全部')
Flags.DEFINE_string('input_dir_HR', None, '推理模式下输入高分辨率数据目录')
Flags.DEFINE_string('mode', 'inference', '训练或推理')
Flags.DEFINE_string('output_dir', None, '模型输出目录')
Flags.DEFINE_string('output_pre', '', '图像子目录名称')
Flags.DEFINE_string('output_name', 'output', '输出文件名前缀')
Flags.DEFINE_string('output_ext', 'jpg', '评估时的输出图片格式')
Flags.DEFINE_string('summary_dir', None, '日志与摘要输出目录')

# Models
Flags.DEFINE_string('checkpoint', None, '若提供，则从该检查点恢复权重')
Flags.DEFINE_integer('num_resblock', 16, '生成器中残差块数量')
# Models for training
Flags.DEFINE_boolean('pre_trained_model', False, '若为 True，则加载生成器权重作为初始点；若为 False，则继续训练')
Flags.DEFINE_string('vgg_ckpt', None, 'VGG19 检查点文件路径')

# Machine resources
Flags.DEFINE_string('cudaID', '0', 'CUDA 设备号')
Flags.DEFINE_integer('queue_thread', 6, '队列线程数（更多线程可加速训练过程')
Flags.DEFINE_integer('name_video_queue_capacity', 512, '文件名队列容量（建议较大以确保足够随机打乱。')
Flags.DEFINE_integer('video_queue_capacity', 256, '视频队列容量（建议较大以确保足够随机打乱')
Flags.DEFINE_integer('video_queue_batch', 2, 'shuffle_batch 队列容量')
                                                   
# Training details
# The data preparing operation
Flags.DEFINE_integer('RNN_N', 10, 'RNN 循环长度')
Flags.DEFINE_integer('batch_size', 4, '输入批次大小')
Flags.DEFINE_boolean('flip', True, '是否应用随机翻转数据增强')
Flags.DEFINE_boolean('random_crop', True, '是否进行随机裁剪')
Flags.DEFINE_boolean('movingFirstFrame', True, '是否随机使用固定移动的首帧。')
Flags.DEFINE_integer('crop_size', 32, '训练图像裁剪尺寸')
# Training data settings
Flags.DEFINE_string('input_video_dir', '', '训练时的视频输入数据目录')
Flags.DEFINE_string('input_video_pre', 'scene', '视频输入数据目录前缀')
Flags.DEFINE_integer('str_dir', 1000, '视频目录起始索引')
Flags.DEFINE_integer('end_dir', 2000, '视频目录结束索引')
Flags.DEFINE_integer('end_dir_val', 2050, '验证视频目录结束索引')
Flags.DEFINE_integer('max_frm', 119, '视频目录最大帧数')
# The loss parameters
Flags.DEFINE_float('vgg_scaling', -0.002, 'VGG 感知损失缩放因子，负值表示禁用')
Flags.DEFINE_float('warp_scaling', 1.0, '光流变形缩放因子')
Flags.DEFINE_boolean('pingpang', False, '是否使用双向循环')
Flags.DEFINE_float('pp_scaling', 1.0, 'PingPang 项因子，仅在 pingpang 为 True 时生效')
# Training parameters
Flags.DEFINE_float('EPS', 1e-12, '防止 NaN 的 epsilon')
Flags.DEFINE_float('learning_rate', 0.0001, '网络学习率')
Flags.DEFINE_integer('decay_step', 500000, '学习率衰减所需步数')
Flags.DEFINE_float('decay_rate', 0.5, '每步衰减率')
Flags.DEFINE_boolean('stair', False, '是否执行阶梯式衰减。True => 在离散区间内衰减。')
Flags.DEFINE_float('beta', 0.9, 'Adam 优化器的 beta1 参数')
Flags.DEFINE_float('adameps', 1e-8, 'Adam 优化器的 eps 参数')
Flags.DEFINE_integer('max_epoch', None, '训练的最大 epoch 数')
Flags.DEFINE_integer('max_iter', 1000000, '训练最大迭代次数')
Flags.DEFINE_integer('display_freq', 20, '训练过程显示频率')
Flags.DEFINE_integer('summary_freq', 100, '写入摘要的频率')
Flags.DEFINE_integer('save_freq', 10000, '保存图像的频率')
# Dst parameters
Flags.DEFINE_float('ratio', 0.01, '内容损失与对抗损失的比率')
Flags.DEFINE_boolean('Dt_mergeDs', True, '是否仅使用合并后的判别器。')
Flags.DEFINE_float('Dt_ratio_0', 1.0, '时间对抗损失初始比率')
Flags.DEFINE_float('Dt_ratio_add', 0.0, '时间对抗损失递增比率')
Flags.DEFINE_float('Dt_ratio_max', 1.0, '时间对抗损失最大比率')
Flags.DEFINE_float('Dbalance', 0.4, '判别器自适应平衡参数')
Flags.DEFINE_float('crop_dt', 0.75, 'dt 裁剪因子') # dt input size = crop_size*crop_dt
Flags.DEFINE_boolean('D_LAYERLOSS', True, '是否使用判别器的层损失')

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
    raise ValueError('需要输出目录')
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
    print("作用域 %s：" % scope)
    variables_names = [ [v.name, v.get_shape().as_list()] for v in tf.get_collection(key, scope=scope)]
    total_sz = 0
    for k in variables_names:
        print ("变量: " + k[0])
        print ("形状: " + str(k[1]))
        total_sz += np.prod(k[1])
    print("参数总量：%d" %total_sz)
    
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
    print('[训练中测试] 第 %d 步：'%testno)
    print(' '.join(cmd1))
    # ignore signals
    return subprocess.Popen(cmd1, preexec_fn = preexec)
    
if False: # If you want to take a look of the configuration, True
    print_configuration_op(FLAGS)

# the inference mode (just perform super resolution on the input image)
if FLAGS.mode == 'inference':
    if FLAGS.checkpoint is None:
        raise ValueError('执行测试需要检查点文件。')

    # Declare the test data reader
    inference_data = inference_data_loader(FLAGS)
    input_shape = [1,] + list(inference_data.inputs[0].shape)
    output_shape = [1,input_shape[1]*4, input_shape[2]*4, 3]
    oh = input_shape[1] - input_shape[1]//8 * 8
    ow = input_shape[2] - input_shape[2]//8 * 8
    paddings = tf.constant([[0,0], [0,oh], [0,ow], [0,0]])
    print("输入形状：", input_shape)
    print("输出形状：", output_shape)
    
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

    print('网络构建完成')
    
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
        
        print('正在从检查点模型加载权重')
        weight_initiallizer.restore(sess, FLAGS.checkpoint)
        if False: # If you want to take a look of the weights, True
            printVariable('generator')
            printVariable('fnet')
        max_iter = len(inference_data.inputs)
                
        srtime = 0
        print('开始帧评估！！')
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
                print('保存图像 %s' % filename)
                out_path = os.path.join(image_dir, "%s.%s"%(filename,FLAGS.output_ext))
                save_img(out_path, output_frame[0])
            else:# First 5 is a hard-coded symmetric frame padding, ignored but time added!
                print("预热中，还需 %d 帧"%(5-i))
    print("总耗时 " + str(srtime) + "，帧数 " + str(max_iter))
        
# The training mode
elif FLAGS.mode == 'train':
    # hard coded save
    filelist = ['main.py','lib/Teco.py','lib/frvsr.py','lib/dataloader.py','lib/ops.py']
    for filename in filelist:
        shutil.copyfile('./' + filename, FLAGS.summary_dir + filename.replace("/","_"))
        
    useValidat = tf.placeholder_with_default( tf.constant(False, dtype=tf.bool), shape=() )
    rdata = frvsr_gpu_data_loader(FLAGS, useValidat)
    # Data = collections.namedtuple('Data', 'paths_HR, s_inputs, s_targets, image_count, steps_per_epoch')
    print('训练数据数量 = %d, 每轮步数 %d' % (rdata.image_count, rdata.steps_per_epoch))
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
        print('准备从预训练模型加载 %d 个生成器和 fnet 权重'%len(assign_ops))
        if FLAGS.ratio>0:
            model_var_list = tf.get_collection(tfflag, scope='tdiscriminator')
            dis_list = get_existing_from_ckpt(FLAGS.checkpoint, model_var_list, print_level=0)
            print('准备从预训练模型加载 %d 个判别器权重'%len(dis_list))
            assign_ops += dis_list
        
    if FLAGS.vgg_scaling > 0.0: # VGG weights are not trainable
        vgg_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg_19')
        vgg_restore = tf.train.Saver(vgg_var_list)
    
    print('网络构建完成。')
    
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
            print('VGG19 恢复成功！！')
                
        if (FLAGS.checkpoint is not None):
            if (FLAGS.pre_trained_model is False):
                print('正在从检查点加载全部内容以继续训练...')
                saver.restore(sess, FLAGS.checkpoint)
                # this will restore everything, including ADAM training parameters and global_step
            else:
                print('正在从预训练模型加载权重以开始新训练...')
                sess.run(assign_ops) # only restore existing model weights
            
        print('第一次运行会花费较长时间来加载训练数据...')
        # get the session for save
        _sess = sess
        while type(_sess).__name__ != 'Session':
            # pylint: disable=W0212
            _sess = _sess._sess
        save_sess = _sess

        if 1:
            print('在训练前保存初始检查点')
            init_run_no = sess.run(Net.global_step)
            saver.save(save_sess, os.path.join(FLAGS.output_dir, 'model'), global_step=init_run_no)
            testWhileTrain(FLAGS, init_run_no) # make sure that testWhileTrain works
        
        # Performing the training
        frame_len = (FLAGS.RNN_N*2-1) if FLAGS.pingpang else FLAGS.RNN_N
        max_iter, step, start = FLAGS.max_iter, 0, time.time()
        if max_iter is None:
            if FLAGS.max_epoch is None:
                    raise ValueError('必须提供 max_epoch 或 max_iter 中的一个')
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
                    print('优化开始！！！(按 Ctrl+C 停止，将尝试保存最后一个模型...)')
                
                if (run_step % FLAGS.summary_freq) == 0:
                    print('正在记录摘要！！')
                    train_writer.add_summary(results['summary'], run_step)
                    val_fetches = {}
                    for name, value in zip(Net.update_list_name[:uplen], Net.update_list):
                        val_fetches['val_' + name] = value
                    val_fetches['summary'] = val_merged
                    val_results = sess.run(val_fetches, feed_dict={useValidat: True})
                    train_writer.add_summary(val_results['summary'], run_step)
                    print('-----------验证数据标量-----------')
                    for name in Net.update_list_name[:uplen]:
                        print('val_' + name, val_results['val_' + name])
                        
                if (run_step % FLAGS.display_freq) == 0:
                    train_epoch = math.ceil(run_step / rdata.steps_per_epoch)
                    train_step = (run_step - 1) % rdata.steps_per_epoch + 1
                    rate = (step + 1) * FLAGS.batch_size / (time.time() - start)
                    remaining = (max_iter - step) * FLAGS.batch_size / rate
                    print("进度  轮次 %d  步骤 %d  图像/秒 %0.1fx%02d  剩余 %d 小时 %d 分钟" % 
                        (train_epoch, train_step, rate, frame_len, 
                        remaining // 3600, (remaining%3600) // 60))
                        
                    print("全局步数", run_step)
                    print("学习率", results['learning_rate'])
                    for name in Net.update_list_name:
                        print(name, results[name])
                        
                if (run_step % FLAGS.save_freq) == 0:
                    print('保存检查点')
                    saver.save(save_sess, os.path.join(FLAGS.output_dir, 'model'), global_step=int(run_step))
                    testWhileTrain(FLAGS, run_step)
                
        except KeyboardInterrupt:
            if step > 1:
                print('main.py: KeyboardInterrupt->正在保存检查点')
                saver.save(save_sess, os.path.join(FLAGS.output_dir, 'model'), global_step=int(run_step))
                testWhileTrain(FLAGS, run_step).communicate()
            print('main.py: 退出')
            exit()
        print('优化完成!!!!!!!!!!!!')
