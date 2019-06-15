'''
several running examples, run with
python3 runGan.py 1 # the last number is the run case number

runcase == 1    inference a trained model
runcase == 2    calculate the metrics, and save the numbers in csv
runcase == 3    training TecoGAN
runcase == 4    training FRVSR
runcase == ...  coming... data preparation and so on...
'''
import os, subprocess, sys, datetime, signal, shutil

runcase = int(sys.argv[1])
print ("Testing test case %d" % runcase)

def preexec(): # Don't forward signals.
    os.setpgrp()
    
def mycall(cmd, block=False):
    if not block:
        return subprocess.Popen(cmd)
    else:
        return subprocess.Popen(cmd, preexec_fn = preexec)
    
def folder_check(path):
    try_num = 1
    oripath = path[:-1] if path.endswith('/') else path
    while os.path.exists(path):
        print("Delete existing folder " + path + "?(Y/N)")
        decision = input()
        if decision == "Y":
            shutil.rmtree(path, ignore_errors=True)
            break
        else:
            path = oripath + "_%d/"%try_num
            try_num += 1
            print(path)
    
    return path

if( runcase == 0 ): # download inference data, trained models
    # download the trained model
    if(not os.path.exists("./model/")): os.mkdir("./model/")
    cmd1 = "wget https://ge.in.tum.de/download/data/TecoGAN/model.zip -O model/model.zip;"
    cmd1 += "unzip model/model.zip -d model; rm model/model.zip"
    subprocess.call(cmd1, shell=True)
    
    # download some test data
    cmd2 = "wget https://ge.in.tum.de/download/data/TecoGAN/vid3_LR.zip -O LR/vid3.zip;"
    cmd2 += "unzip LR/vid3.zip -d LR; rm LR/vid3.zip"
    subprocess.call(cmd2, shell=True)
    
    cmd2 = "wget https://ge.in.tum.de/download/data/TecoGAN/tos_LR.zip -O LR/tos.zip;"
    cmd2 += "unzip LR/tos.zip -d LR; rm LR/tos.zip"
    subprocess.call(cmd2, shell=True)
    
    # download the ground-truth data
    if(not os.path.exists("./HR/")): os.mkdir("./HR/")
    cmd3 = "wget https://ge.in.tum.de/download/data/TecoGAN/vid4_HR.zip -O HR/vid4.zip;"
    cmd3 += "unzip HR/vid4.zip -d HR; rm HR/vid4.zip"
    subprocess.call(cmd3, shell=True)
    
    cmd3 = "wget https://ge.in.tum.de/download/data/TecoGAN/tos_HR.zip -O HR/tos.zip;"
    cmd3 += "unzip HR/tos.zip -d HR; rm HR/tos.zip"
    subprocess.call(cmd3, shell=True)
    
elif( runcase == 1 ): # inference a trained model
    
    dirstr = './results/' # the place to save the results
    testpre = ['calendar'] # the test cases

    if (not os.path.exists(dirstr)): os.mkdir(dirstr)
    
    # run these test cases one by one:
    for nn in range(len(testpre)):
        cmd1 = ["python3", "main.py",
            "--cudaID", "0",            # set the cudaID here to use only one GPU
            "--output_dir",  dirstr,    # Set the place to put the results.
            "--summary_dir", os.path.join(dirstr, 'log/'), # Set the place to put the log. 
            "--mode","inference", 
            "--input_dir_LR", os.path.join("./LR/", testpre[nn]),   # the LR directory
            #"--input_dir_HR", os.path.join("./HR/", testpre[nn]),  # the HR directory
            # one of (input_dir_HR,input_dir_LR) should be given
            "--output_pre", testpre[nn], # the subfolder to save current scene, optional
            "--num_resblock", "16",  # our model has 16 residual blocks, 
            # the pre-trained FRVSR and TecoGAN mini have 10 residual blocks
            "--checkpoint", './model/TecoGAN',  # the path of the trained model,
            "--output_ext", "png"               # png is more accurate, jpg is smaller
        ]
        mycall(cmd1).communicate()

elif( runcase == 2 ): # calculate all metrics, and save the csv files, should use png

    testpre = ["calendar"] # just put more scenes to evaluate all of them
    dirstr = './results/'  # the outputs
    tarstr = './HR/'       # the GT

    tar_list = [(tarstr+_) for _ in testpre]
    out_list = [(dirstr+_) for _ in testpre]
    cmd1 = ["python3", "metrics.py",
        "--output", dirstr+"metric_log/",
        "--results", ",".join(out_list),
        "--targets", ",".join(tar_list),
    ]
    mycall(cmd1).communicate()
    
elif( runcase == 3 ): # Train TecoGAN
    '''
    In order to use the VGG as a perceptual loss,
    we download from TensorFlow-Slim image classification model library:
    https://github.com/tensorflow/models/tree/master/research/slim    
    '''
    VGGPath = "model/" # the path for the VGG model, there should be a vgg_19.ckpt inside
    VGGModelPath = os.path.join(VGGPath, "vgg_19.ckpt")
    if(not os.path.exists(VGGPath)): os.mkdir(VGGPath)
    if(not os.path.exists(VGGModelPath)):
        # Download the VGG 19 model from 
        print("VGG model not found, downloading to %s"%VGGPath)
        cmd0 = "wget http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz -O " + os.path.join(VGGPath, "vgg19.tar.gz")
        cmd0 += ";tar -xvf " + os.path.join(VGGPath,"vgg19.tar.gz") + " -C " + VGGPath + "; rm "+ os.path.join(VGGPath, "vgg19.tar.gz")
        subprocess.call(cmd0, shell=True)
        
    '''
    Use our pre-trained FRVSR model. If you want to train one, try runcase 4, and update this path by:
    FRVSRModel = "ex_FRVSRmm-dd-hh/model-500000"
    '''
    FRVSRModel = "model/ourFRVSR" 
    if(not os.path.exists(FRVSRModel+".data-00000-of-00001")):
        # Download our pre-trained FRVSR model
        print("pre-trained FRVSR model not found, downloading")
        cmd0 = "wget http://ge.in.tum.de/download/2019-TecoGAN/FRVSR_Ours.zip -O model/ofrvsr.zip;"
        cmd0 += "unzip model/ofrvsr.zip -d model; rm model/ofrvsr.zip"
        subprocess.call(cmd0, shell=True)
    
    TrainingDataPath = "/mnt/netdisk/video_data/" 
    
    '''Prepare Training Folder'''
    # path appendix, manually define it, or use the current datetime, now_str = "mm-dd-hh"
    now_str = datetime.datetime.now().strftime("%m-%d-%H")
    train_dir = folder_check("ex_TecoGAN%s/"%now_str)
    # train TecoGAN, loss = l2 + VGG54 loss + A spatio-temporal Discriminator
    cmd1 = ["python3", "main.py",
        "--cudaID", "0", # set the cudaID here to use only one GPU
        "--output_dir", train_dir, # Set the place to save the models.
        "--summary_dir", os.path.join(train_dir,"log/"), # Set the place to save the log. 
        "--mode","train",
        "--batch_size", "4" , # small, because GPU memory is not big
        "--RNN_N", "10" , # train with a sequence of RNN_N frames, >6 is better, >10 is not necessary
        "--movingFirstFrame", # a data augmentation
        "--random_crop",
        "--crop_size", "32",
        "--learning_rate", "0.00005",
        # -- learning_rate step decay, here it is not used --
        "--decay_step", "500000", 
        "--decay_rate", "1.0", # 1.0 means no decay
        "--stair",
        "--beta", "0.9", # ADAM training parameter beta
        "--max_iter", "500000", # 500k or more, the one we present is trained for 900k
        "--save_freq", "10000", # the frequency we save models
        # -- network architecture parameters --
        "--num_resblock", "16", # FRVSR and TecoGANmini has num_resblock as 10. The TecoGAN has 16.
        # -- VGG loss, disable with vgg_scaling < 0
        "--vgg_scaling", "0.2",
        "--vgg_ckpt", VGGModelPath, # necessary if vgg_scaling > 0
    ]
    '''Video Training data:
    please udate the TrainingDataPath according to ReadMe.md
    input_video_pre is hard coded as scene in dataPrepare.py at line 142
    str_dir is the starting index for training data
    end_dir is the ending index for training data
    end_dir+1 is the starting index for validation data
    end_dir_val is the ending index for validation data
    max_frm should be duration (in dataPrepare.py) -1
    queue_thread: how many cpu can be used for loading data when training
    name_video_queue_capacity, video_queue_capacity: how much memory can be used
    '''
    cmd1 += [
        "--input_video_dir", TrainingDataPath, 
        "--input_video_pre", "scene",
        "--str_dir", "2000",
        "--end_dir", "2250",
        "--end_dir_val", "2290",
        "--max_frm", "119",
        # -- cpu memory for data loading --
        "--queue_thread", "12",# Cpu threads for the data. >4 to speedup the training
        "--name_video_queue_capacity", "1024",
        "--video_queue_capacity", "1024",
    ]
    '''
    loading the pre-trained model from FRVSR can make the training faster
    --checkpoint, path of the model, here our pre-trained FRVSR is given
    --pre_trained_model,  to continue an old (maybe accidentally stopeed) training, 
        pre_trained_model should be false, and checkpoint should be the last model such as 
        ex_TecoGANmm-dd-hh/model-xxxxxxx
        To start a new and different training, pre_trained_model is True. 
        The difference here is 
        whether to load the whole graph icluding ADAM training averages/momentums/ and so on
        or just load existing pre-trained weights.
    '''
    cmd1 += [ # based on a pre-trained FRVSR model. Here we want to train a new adversarial training
        "--pre_trained_model", # True
        "--checkpoint", FRVSRModel,
    ]
    
    # the following can be used to train TecoGAN continuously
    # old_model = "model/ex_TecoGANmm-dd-hh/model-xxxxxxx" 
    # cmd1 += [ # Here we want to train continuously
    #     "--nopre_trained_model", # False
    #     "--checkpoint", old_model,
    # ]
    
    ''' parameters for GAN training '''
    cmd1 += [
        "--ratio", "0.01",  # the ratio for the adversarial loss from the Discriminator to the Generator
        "--Dt_mergeDs",     # if Dt_mergeDs == False, only use temporal inputs, so we have a temporal Discriminator
                            # else, use both temporal and spatial inputs, then we have a Dst, the spatial and temporal Discriminator
    ]
    ''' if the generator is pre-trained, to fade in the discriminator is usually more stable.
    the weight of the adversarial loss will be weighed with a weight, started from Dt_ratio_0, 
    and increases until Dt_ratio_max, the increased value is Dt_ratio_add per training step
    For example, fading Dst in smoothly in the first 4k steps is 
    "--Dt_ratio_max", "1.0", "--Dt_ratio_0", "0.0", "--Dt_ratio_add", "0.00025"
    '''
    cmd1 += [ # here, the fading in is disabled 
        "--Dt_ratio_max", "1.0",
        "--Dt_ratio_0", "1.0", 
        "--Dt_ratio_add", "0.0", 
    ]
    ''' Other Losses '''
    cmd1 += [
        "--pingpang",           # our Ping-Pang loss
        "--pp_scaling", "0.5",  # the weight of the our bi-directional loss, 0.0~0.5
        "--D_LAYERLOSS",        # use feature layer losses from the discriminator
    ]
    
    pid = mycall(cmd1, block=True) 
    try: # catch interruption for training
        pid.communicate()
    except KeyboardInterrupt: # Ctrl + C to stop current training try to save the last model 
        print("runGAN.py: sending SIGINT signal to the sub process...")
        pid.send_signal(signal.SIGINT)
        # try to save the last model 
        pid.communicate()
        print("runGAN.py: finished...")
        
        
elif( runcase == 4 ): # Train FRVSR, loss = l2 warp + l2 content
    now_str = datetime.datetime.now().strftime("%m-%d-%H")
    train_dir = folder_check("ex_FRVSR%s/"%now_str)
    cmd1 = ["python3", "main.py",
        "--cudaID", "0", # set the cudaID here to use only one GPU
        "--output_dir", train_dir, # Set the place to save the models.
        "--summary_dir", os.path.join(train_dir,"log/"), # Set the place to save the log. 
        "--mode","train",
        "--batch_size", "4" , # small, because GPU memory is not big
        "--RNN_N", "10" , # train with a sequence of RNN_N frames, >6 is better, >10 is not necessary
        "--movingFirstFrame", # a data augmentation
        "--random_crop",
        "--crop_size", "32",
        "--learning_rate", "0.00005",
        # -- learning_rate step decay, here it is not used --
        "--decay_step", "500000", 
        "--decay_rate", "1.0", # 1.0 means no decay
        "--stair",
        "--beta", "0.9", # ADAM training parameter beta
        "--max_iter", "500000", # 500k is usually fine for FRVSR, GAN versions need more to be stable
        "--save_freq", "10000", # the frequency we save models
        # -- network architecture parameters --
        "--num_resblock", "10", # a smaller model
        "--ratio", "-0.01",  # the ratio for the adversarial loss, negative means disabled
        "--nopingpang",
    ]
    '''Video Training data... Same as runcase 3...'''
    TrainingDataPath = "/mnt/netdisk/video_data/"
    cmd1 += [
        "--input_video_dir", TrainingDataPath, 
        "--input_video_pre", "scene",
        "--str_dir", "2000",
        "--end_dir", "2250",
        "--end_dir_val", "2290",
        "--max_frm", "119",
        # -- cpu memory for data loading --
        "--queue_thread", "12",# Cpu threads for the data. >4 to speedup the training
        "--name_video_queue_capacity", "1024",
        "--video_queue_capacity", "1024",
    ]
    
    pid = mycall(cmd1, block=True)
    try: # catch interruption for training
        pid.communicate()
    except KeyboardInterrupt: # Ctrl + C to stop current training try to save the last model 
        print("runGAN.py: sending SIGINT signal to the sub process...")
        pid.send_signal(signal.SIGINT)
        # try to save the last model 
        pid.communicate()
        print("runGAN.py: finished...")