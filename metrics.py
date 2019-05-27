import numpy as np
import cv2
import os, sys
import pandas as pd
from LPIPSmodels import util
import LPIPSmodels.dist_model as dm
from skimage.measure import compare_ssim

from absl import flags
flags.DEFINE_string('output', None, 'the path of output directory')
flags.DEFINE_string('results', None, 'the list of paths of result directory')
flags.DEFINE_string('targets', None, 'the list of paths of target directory')

FLAGS = flags.FLAGS
FLAGS(sys.argv)

if(not os.path.exists(FLAGS.output)):
    os.mkdir(FLAGS.output)
    
# The operation used to print out the configuration
def print_configuration_op(FLAGS):
    print('[Configurations]:')
    for name, value in FLAGS.flag_values_dict().items():
        print('\t%s: %s'%(name, str(value)))
    print('End of configuration')
# custom Logger to write Log to file

def listPNGinDir(dirpath):
    filelist = os.listdir(dirpath)
    filelist = [_ for _ in filelist if _.endswith(".png")] 
    filelist = [_ for _ in filelist if not _.startswith("IB")] 
    filelist = sorted(filelist)
    filelist.sort(key=lambda f: int(''.join(list(filter(str.isdigit, f))) or -1))
    result = [os.path.join(dirpath,_) for _ in filelist if _.endswith(".png")]
    return result

def _rgb2ycbcr(img, maxVal=255):
##### color space transform, originally from https://github.com/yhjo09/VSR-DUF ##### 
    O = np.array([[16],
                  [128],
                  [128]])
    T = np.array([[0.256788235294118, 0.504129411764706, 0.097905882352941],
                  [-0.148223529411765, -0.290992156862745, 0.439215686274510],
                  [0.439215686274510, -0.367788235294118, -0.071427450980392]])

    if maxVal == 1:
        O = O / 255.0

    t = np.reshape(img, (img.shape[0]*img.shape[1], img.shape[2]))
    t = np.dot(t, np.transpose(T))
    t[:, 0] += O[0]
    t[:, 1] += O[1]
    t[:, 2] += O[2]
    ycbcr = np.reshape(t, [img.shape[0], img.shape[1], img.shape[2]])

    return ycbcr

def to_uint8(x, vmin, vmax):
##### color space transform, originally from https://github.com/yhjo09/VSR-DUF ##### 
    x = x.astype('float32')
    x = (x-vmin)/(vmax-vmin)*255 # 0~255
    return np.clip(np.round(x), 0, 255)

def psnr(img_true, img_pred):
##### PSNR with color space transform, originally from https://github.com/yhjo09/VSR-DUF ##### 
    Y_true = _rgb2ycbcr(to_uint8(img_true, 0, 255), 255)[:,:,0]
    Y_pred = _rgb2ycbcr(to_uint8(img_pred, 0, 255), 255)[:,:,0]
    diff =  Y_true - Y_pred
    rmse = np.sqrt(np.mean(np.power(diff,2)))
    return 20*np.log10(255./rmse)
    
def ssim(img_true, img_pred): ##### SSIM ##### 
    Y_true = _rgb2ycbcr(to_uint8(img_true, 0, 255), 255)[:,:,0]
    Y_pred = _rgb2ycbcr(to_uint8(img_pred, 0, 255), 255)[:,:,0]
    return compare_ssim(Y_true, Y_pred, data_range=Y_pred.max() - Y_pred.min())

def crop_8x8( img ):
    ori_h = img.shape[0]
    ori_w = img.shape[1]
    
    h = (ori_h//32) * 32
    w = (ori_w//32) * 32
    
    while(h > ori_h - 16):
        h = h - 32
    while(w > ori_w - 16):
        w = w - 32
    
    y = (ori_h - h) // 2
    x = (ori_w - w) // 2
    crop_img = img[y:y+h, x:x+w]
    return crop_img, y, x

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        filename = "metricsfile.txt"
        self.log = open(os.path.join(FLAGS.output, filename), "a") 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message) 
    def flush(self):
        self.log.flush()
        
sys.stdout = Logger()

print_configuration_op(FLAGS)

result_list = FLAGS.results.split(',')
target_list = FLAGS.targets.split(',')
folder_n = len(result_list)


model = dm.DistModel()
model.initialize(model='net-lin',net='alex',use_gpu=True)

cutfr = 2
# maxV = 0.4, for line 154-166

keys = ["PSNR", "SSIM", "LPIPS", "tOF", "tLP100"] # keys = ["LPIPS"]
sum_dict    = dict.fromkeys(["FrameAvg_"+_ for _ in keys], 0)
len_dict    = dict.fromkeys(keys, 0)
avg_dict    = dict.fromkeys(["Avg_"+_ for _ in keys], 0)
folder_dict = dict.fromkeys(["FolderAvg_"+_ for _ in keys], 0)

for folder_i in range(folder_n):
    result = listPNGinDir(result_list[folder_i])
    target = listPNGinDir(target_list[folder_i])
    image_no = len(target)
    
    list_dict = {}
    for key_i in keys:
        list_dict[key_i] = []
    
    for i in range(cutfr, image_no-cutfr):
        output_img = cv2.imread(result[i])[:,:,::-1]
        target_img = cv2.imread(target[i])[:,:,::-1]
        msg = "frame %d, tar %s, out %s, "%(i, str(target_img.shape), str(output_img.shape))
        if( target_img.shape[0] < output_img.shape[0]) or ( target_img.shape[1] < output_img.shape[1]): # target is not dividable by 4
            output_img = output_img[:target_img.shape[0],:target_img.shape[1]]
        print(result[i])
        
        if "tOF" in keys:# tOF
            output_grey = cv2.cvtColor(output_img, cv2.COLOR_RGB2GRAY)
            target_grey = cv2.cvtColor(target_img, cv2.COLOR_RGB2GRAY)
            if (i > cutfr): # temporal metrics
                target_OF=cv2.calcOpticalFlowFarneback(pre_tar_grey, target_grey, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                output_OF=cv2.calcOpticalFlowFarneback(pre_out_grey, output_grey, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                target_OF, ofy, ofx = crop_8x8(target_OF)
                output_OF, ofy, ofx = crop_8x8(output_OF)
                OF_diff = np.absolute(target_OF - output_OF)
                if False: # for motion visualization
                    tOFpath = os.path.join(FLAGS.output,"%03d_tOF"%folder_i)
                    if(not os.path.exists(tOFpath)): os.mkdir(tOFpath)
                    hsv = np.zeros_like(output_img)
                    hsv[...,1] = 255
                    out_path = os.path.join(tOFpath, "flow_%04d.jpg" %i)
                    mag, ang = cv2.cartToPolar(OF_diff[...,0], OF_diff[...,1])
                    # print("tar max %02.6f, min %02.6f, avg %02.6f" % (mag.max(), mag.min(), mag.mean()))
                    mag = np.clip(mag, 0.0, maxV)/maxV
                    hsv[...,0] = ang*180/np.pi/2
                    hsv[...,2] = mag * 255.0 #
                    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
                    cv2.imwrite(out_path, bgr)
                    
                OF_diff = np.sqrt(np.sum(OF_diff * OF_diff, axis = -1)) # l1 vector norm
                # OF_diff, ofy, ofx = crop_8x8(OF_diff)
                list_dict["tOF"].append( OF_diff.mean() )
                msg += "tOF %02.2f, " %(list_dict["tOF"][-1])
            
            pre_out_grey = output_grey
            pre_tar_grey = target_grey

        target_img, ofy, ofx = crop_8x8(target_img)
        output_img, ofy, ofx = crop_8x8(output_img)
            
        if "PSNR" in keys:# psnr
            list_dict["PSNR"].append( psnr(target_img, output_img) )
            msg +="psnr %02.2f" %(list_dict["PSNR"][-1])
        
        if "SSIM" in keys:# ssim
            list_dict["SSIM"].append( ssim(target_img, output_img) )
            msg +=", ssim %02.2f" %(list_dict["SSIM"][-1])
            
        if "LPIPS" in keys or "tLP100" in keys:
            img0 = util.im2tensor(target_img) # RGB image from [-1,1]
            img1 = util.im2tensor(output_img)
        
            if "LPIPS" in keys: # LPIPS
                dist01 = model.forward(img0,img1)
                list_dict["LPIPS"].append( dist01[0] )
                msg +=", lpips %02.2f" %(dist01[0])
            
            if "tLP100" in keys and (i > cutfr):# tLP, temporal metrics
                dist0t = model.forward(pre_img0, img0)
                dist1t = model.forward(pre_img1, img1)
                # print ("tardis %f, outdis %f" %(dist0t, dist1t))
                dist01t = np.absolute(dist0t - dist1t) * 100.0 ##########!!!!!
                list_dict["tLP100"].append( dist01t[0] )
                msg += ", tLPx100 %02.2f" %(dist01t[0])
            pre_img0 = img0
            pre_img1 = img1
        
        msg +=", crop (%d, %d)" %(ofy, ofx)
        print(msg)
    mode = 'w' if folder_i==0 else 'a'
    
    pd_dict = {}
    for cur_num_data in keys:
        num_data = cur_num_data+"_%02d" % folder_i
        cur_list = np.float32(list_dict[cur_num_data])
        pd_dict[num_data] = pd.Series(cur_list)
        
        num_data_sum = cur_list.sum()
        num_data_len = cur_list.shape[0]
        num_data_mean = num_data_sum / num_data_len
        print("%s, max %02.4f, min %02.4f, avg %02.4f" % 
            (num_data, cur_list.max(), cur_list.min(), num_data_mean))
            
        if folder_i == 0:
            avg_dict["Avg_"+cur_num_data] = [num_data_mean]
        else:
            avg_dict["Avg_"+cur_num_data] += [num_data_mean]
        
        sum_dict["FrameAvg_"+cur_num_data] += num_data_sum
        len_dict[cur_num_data] += num_data_len
        folder_dict["FolderAvg_"+cur_num_data] += num_data_mean
        
    pd.DataFrame(pd_dict).to_csv(os.path.join(FLAGS.output,"metrics.csv"), mode=mode)
    
for num_data in keys:
    sum_dict["FrameAvg_"+num_data] = pd.Series([sum_dict["FrameAvg_"+num_data] / len_dict[num_data]])
    folder_dict["FolderAvg_"+num_data] = pd.Series([folder_dict["FolderAvg_"+num_data] / folder_n])
    avg_dict["Avg_"+num_data] = pd.Series(np.float32(avg_dict["Avg_"+num_data]))
    print("%s, total frame %d, total avg %02.4f, folder avg %02.4f" % 
        (num_data, len_dict[num_data], sum_dict["FrameAvg_"+num_data][0], folder_dict["FolderAvg_"+num_data][0]))
pd.DataFrame(avg_dict).to_csv(os.path.join(FLAGS.output,"metrics.csv"), mode='a')
pd.DataFrame(folder_dict).to_csv(os.path.join(FLAGS.output,"metrics.csv"), mode='a')
pd.DataFrame(sum_dict).to_csv(os.path.join(FLAGS.output,"metrics.csv"), mode='a')
print("Finished.")