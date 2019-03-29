
from __future__ import absolute_import

import sys
sys.path.append('..')
sys.path.append('.')
import numpy as np
import torch
from torch import nn
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
from .base_model import BaseModel
from scipy.ndimage import zoom
import fractions
import functools
import skimage.transform
from IPython import embed

from . import networks_basic as networks
from . import util

class DistModel(BaseModel):
    def name(self):
        return self.model_name

    def initialize(self, model='net-lin', net='alex', pnet_rand=False, pnet_tune=False, model_path=None, colorspace='Lab', use_gpu=True, printNet=False, spatial=False, spatial_shape=None, spatial_order=1, spatial_factor=None, is_train=False, lr=.0001, beta1=0.5, version='0.1'):
        '''
        INPUTS
            model - ['net-lin'] for linearly calibrated network
                    ['net'] for off-the-shelf network
                    ['L2'] for L2 distance in Lab colorspace
                    ['SSIM'] for ssim in RGB colorspace
            net - ['squeeze','alex','vgg']
            model_path - if None, will look in weights/[NET_NAME].pth
            colorspace - ['Lab','RGB'] colorspace to use for L2 and SSIM
            use_gpu - bool - whether or not to use a GPU
            printNet - bool - whether or not to print network architecture out
            spatial - bool - whether to output an array containing varying distances across spatial dimensions
            spatial_shape - if given, output spatial shape. if None then spatial shape is determined automatically via spatial_factor (see below).
            spatial_factor - if given, specifies upsampling factor relative to the largest spatial extent of a convolutional layer. if None then resized to size of input images.
            spatial_order - spline order of filter for upsampling in spatial mode, by default 1 (bilinear).
            is_train - bool - [True] for training mode
            lr - float - initial learning rate
            beta1 - float - initial momentum term for adam
            version - 0.1 for latest, 0.0 was original
        '''
        BaseModel.initialize(self, use_gpu=use_gpu)

        self.model = model
        self.net = net
        self.use_gpu = use_gpu
        self.is_train = is_train
        self.spatial = spatial
        self.spatial_shape = spatial_shape
        self.spatial_order = spatial_order
        self.spatial_factor = spatial_factor

        self.model_name = '%s [%s]'%(model,net)
        if(self.model == 'net-lin'): # pretrained net + linear layer
            self.net = networks.PNetLin(use_gpu=use_gpu,pnet_rand=pnet_rand, pnet_tune=pnet_tune, pnet_type=net,use_dropout=True,spatial=spatial,version=version)
            kw = {}
            if not use_gpu:
                kw['map_location'] = 'cpu'
            if(model_path is None):
                import inspect
                # model_path = './PerceptualSimilarity/weights/v%s/%s.pth'%(version,net)
                model_path = os.path.abspath(os.path.join(inspect.getfile(self.initialize), '..', 'v%s/%s.pth'%(version,net)))

            if(not is_train):
                print('Loading model from: %s'%model_path)
                self.net.load_state_dict(torch.load(model_path, **kw))

        elif(self.model=='net'): # pretrained network
            assert not self.spatial, 'spatial argument not supported yet for uncalibrated networks'
            self.net = networks.PNet(use_gpu=use_gpu,pnet_type=net)
            self.is_fake_net = True
        elif(self.model in ['L2','l2']):
            self.net = networks.L2(use_gpu=use_gpu,colorspace=colorspace) # not really a network, only for testing
            self.model_name = 'L2'
        elif(self.model in ['DSSIM','dssim','SSIM','ssim']):
            self.net = networks.DSSIM(use_gpu=use_gpu,colorspace=colorspace)
            self.model_name = 'SSIM'
        else:
            raise ValueError("Model [%s] not recognized." % self.model)

        self.parameters = list(self.net.parameters())

        if self.is_train: # training mode
            # extra network on top to go from distances (d0,d1) => predicted human judgment (h*)
            self.rankLoss = networks.BCERankingLoss(use_gpu=use_gpu)
            self.parameters+=self.rankLoss.parameters
            self.lr = lr
            self.old_lr = lr
            self.optimizer_net = torch.optim.Adam(self.parameters, lr=lr, betas=(beta1, 0.999))
        else: # test mode
            self.net.eval()

        if(printNet):
            print('---------- Networks initialized -------------')
            networks.print_network(self.net)
            print('-----------------------------------------------')

    def forward_pair(self,in1,in2,retPerLayer=False):
        if(retPerLayer):
            return self.net.forward(in1,in2, retPerLayer=True)
        else:
            return self.net.forward(in1,in2)

    def forward(self, in0, in1, retNumpy=True):
        ''' Function computes the distance between image patches in0 and in1
        INPUTS
            in0, in1 - torch.Tensor object of shape Nx3xXxY - image patch scaled to [-1,1]
            retNumpy - [False] to return as torch.Tensor, [True] to return as numpy array
        OUTPUT
            computed distances between in0 and in1
        '''

        self.input_ref = in0
        self.input_p0 = in1

        if(self.use_gpu):
            self.input_ref = self.input_ref.cuda()
            self.input_p0 = self.input_p0.cuda()

        self.var_ref = Variable(self.input_ref,requires_grad=True)
        self.var_p0 = Variable(self.input_p0,requires_grad=True)

        self.d0 = self.forward_pair(self.var_ref, self.var_p0)
        self.loss_total = self.d0

        def convert_output(d0):
            if(retNumpy):
                ans = d0.cpu().data.numpy()
                if not self.spatial:
                    ans = ans.flatten()
                else:
                    assert(ans.shape[0] == 1 and len(ans.shape) == 4)
                    return ans[0,...].transpose([1, 2, 0])                  # Reshape to usual numpy image format: (height, width, channels)
                return ans
            else:
                return d0

        if self.spatial:
            L = [convert_output(x) for x in self.d0]
            spatial_shape = self.spatial_shape
            if spatial_shape is None:
                if(self.spatial_factor is None):
                    spatial_shape = (in0.size()[2],in0.size()[3])
                else:
                    spatial_shape = (max([x.shape[0] for x in L])*self.spatial_factor, max([x.shape[1] for x in L])*self.spatial_factor)
            
            L = [skimage.transform.resize(x, spatial_shape, order=self.spatial_order, mode='edge') for x in L]
            
            L = np.mean(np.concatenate(L, 2) * len(L), 2)
            return L
        else:
            return convert_output(self.d0)

    # ***** TRAINING FUNCTIONS *****
    def optimize_parameters(self):
        self.forward_train()
        self.optimizer_net.zero_grad()
        self.backward_train()
        self.optimizer_net.step()
        self.clamp_weights()

    def clamp_weights(self):
        for module in self.net.modules():
            if(hasattr(module, 'weight') and module.kernel_size==(1,1)):
                module.weight.data = torch.clamp(module.weight.data,min=0)

    def set_input(self, data):
        self.input_ref = data['ref']
        self.input_p0 = data['p0']
        self.input_p1 = data['p1']
        self.input_judge = data['judge']

        if(self.use_gpu):
            self.input_ref = self.input_ref.cuda()
            self.input_p0 = self.input_p0.cuda()
            self.input_p1 = self.input_p1.cuda()
            self.input_judge = self.input_judge.cuda()

        self.var_ref = Variable(self.input_ref,requires_grad=True)
        self.var_p0 = Variable(self.input_p0,requires_grad=True)
        self.var_p1 = Variable(self.input_p1,requires_grad=True)

    def forward_train(self): # run forward pass
        self.d0 = self.forward_pair(self.var_ref, self.var_p0)
        self.d1 = self.forward_pair(self.var_ref, self.var_p1)
        self.acc_r = self.compute_accuracy(self.d0,self.d1,self.input_judge)

        # var_judge
        self.var_judge = Variable(1.*self.input_judge).view(self.d0.size())

        self.loss_total = self.rankLoss.forward(self.d0, self.d1, self.var_judge*2.-1.)
        return self.loss_total

    def backward_train(self):
        torch.mean(self.loss_total).backward()

    def compute_accuracy(self,d0,d1,judge):
        ''' d0, d1 are Variables, judge is a Tensor '''
        d1_lt_d0 = (d1<d0).cpu().data.numpy().flatten()
        judge_per = judge.cpu().numpy().flatten()
        return d1_lt_d0*judge_per + (1-d1_lt_d0)*(1-judge_per)

    def get_current_errors(self):
        retDict = OrderedDict([('loss_total', self.loss_total.data.cpu().numpy()),
                            ('acc_r', self.acc_r)])

        for key in retDict.keys():
            retDict[key] = np.mean(retDict[key])

        return retDict

    def get_current_visuals(self):
        zoom_factor = 256/self.var_ref.data.size()[2]

        ref_img = util.tensor2im(self.var_ref.data)
        p0_img = util.tensor2im(self.var_p0.data)
        p1_img = util.tensor2im(self.var_p1.data)

        ref_img_vis = zoom(ref_img,[zoom_factor, zoom_factor, 1],order=0)
        p0_img_vis = zoom(p0_img,[zoom_factor, zoom_factor, 1],order=0)
        p1_img_vis = zoom(p1_img,[zoom_factor, zoom_factor, 1],order=0)

        return OrderedDict([('ref', ref_img_vis),
                            ('p0', p0_img_vis),
                            ('p1', p1_img_vis)])

    def save(self, path, label):
        self.save_network(self.net, path, '', label)
        self.save_network(self.rankLoss.net, path, 'rank', label)

    def update_learning_rate(self,nepoch_decay):
        lrd = self.lr / nepoch_decay
        lr = self.old_lr - lrd

        for param_group in self.optimizer_net.param_groups:
            param_group['lr'] = lr

        print('update lr [%s] decay: %f -> %f' % (type,self.old_lr, lr))
        self.old_lr = lr



def score_2afc_dataset(data_loader,func):
    ''' Function computes Two Alternative Forced Choice (2AFC) score using
        distance function 'func' in dataset 'data_loader'
    INPUTS
        data_loader - CustomDatasetDataLoader object - contains a TwoAFCDataset inside
        func - callable distance function - calling d=func(in0,in1) should take 2
            pytorch tensors with shape Nx3xXxY, and return numpy array of length N
    OUTPUTS
        [0] - 2AFC score in [0,1], fraction of time func agrees with human evaluators
        [1] - dictionary with following elements
            d0s,d1s - N arrays containing distances between reference patch to perturbed patches 
            gts - N array in [0,1], preferred patch selected by human evaluators
                (closer to "0" for left patch p0, "1" for right patch p1,
                "0.6" means 60pct people preferred right patch, 40pct preferred left)
            scores - N array in [0,1], corresponding to what percentage function agreed with humans
    CONSTS
        N - number of test triplets in data_loader
    '''

    d0s = []
    d1s = []
    gts = []

    # bar = pb.ProgressBar(max_value=data_loader.load_data().__len__())
    for (i,data) in enumerate(data_loader.load_data()):
        d0s+=func(data['ref'],data['p0']).tolist()
        d1s+=func(data['ref'],data['p1']).tolist()
        gts+=data['judge'].cpu().numpy().flatten().tolist()
        # bar.update(i)

    d0s = np.array(d0s)
    d1s = np.array(d1s)
    gts = np.array(gts)
    scores = (d0s<d1s)*(1.-gts) + (d1s<d0s)*gts + (d1s==d0s)*.5

    return(np.mean(scores), dict(d0s=d0s,d1s=d1s,gts=gts,scores=scores))

def score_jnd_dataset(data_loader,func):
    ''' Function computes JND score using distance function 'func' in dataset 'data_loader'
    INPUTS
        data_loader - CustomDatasetDataLoader object - contains a JNDDataset inside
        func - callable distance function - calling d=func(in0,in1) should take 2
            pytorch tensors with shape Nx3xXxY, and return numpy array of length N
    OUTPUTS
        [0] - JND score in [0,1], mAP score (area under precision-recall curve)
        [1] - dictionary with following elements
            ds - N array containing distances between two patches shown to human evaluator
            sames - N array containing fraction of people who thought the two patches were identical
    CONSTS
        N - number of test triplets in data_loader
    '''

    ds = []
    gts = []

    # bar = pb.ProgressBar(max_value=data_loader.load_data().__len__())
    for (i,data) in enumerate(data_loader.load_data()):
        ds+=func(data['p0'],data['p1']).tolist()
        gts+=data['same'].cpu().numpy().flatten().tolist()
        # bar.update(i)

    sames = np.array(gts)
    ds = np.array(ds)

    sorted_inds = np.argsort(ds)
    ds_sorted = ds[sorted_inds]
    sames_sorted = sames[sorted_inds]

    TPs = np.cumsum(sames_sorted)
    FPs = np.cumsum(1-sames_sorted)
    FNs = np.sum(sames_sorted)-TPs

    precs = TPs/(TPs+FPs)
    recs = TPs/(TPs+FNs)
    score = util.voc_ap(recs,precs)

    return(score, dict(ds=ds,sames=sames))
