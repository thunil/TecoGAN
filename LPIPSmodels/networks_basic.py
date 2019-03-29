
from __future__ import absolute_import

import sys
sys.path.append('..')
sys.path.append('.')
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
from pdb import set_trace as st
from skimage import color
from IPython import embed
from . import pretrained_networks as pn

from . import util

# Off-the-shelf deep network
class PNet(nn.Module):
    '''Pre-trained network with all channels equally weighted by default'''
    def __init__(self, pnet_type='vgg', pnet_rand=False, use_gpu=True):
        super(PNet, self).__init__()

        self.use_gpu = use_gpu

        self.pnet_type = pnet_type
        self.pnet_rand = pnet_rand

        self.shift = torch.autograd.Variable(torch.Tensor([-.030, -.088, -.188]).view(1,3,1,1))
        self.scale = torch.autograd.Variable(torch.Tensor([.458, .448, .450]).view(1,3,1,1))
        
        if(self.pnet_type in ['vgg','vgg16']):
            self.net = pn.vgg16(pretrained=not self.pnet_rand,requires_grad=False)
        elif(self.pnet_type=='alex'):
            self.net = pn.alexnet(pretrained=not self.pnet_rand,requires_grad=False)
        elif(self.pnet_type[:-2]=='resnet'):
            self.net = pn.resnet(pretrained=not self.pnet_rand,requires_grad=False, num=int(self.pnet_type[-2:]))
        elif(self.pnet_type=='squeeze'):
            self.net = pn.squeezenet(pretrained=not self.pnet_rand,requires_grad=False)

        self.L = self.net.N_slices

        if(use_gpu):
            self.net.cuda()
            self.shift = self.shift.cuda()
            self.scale = self.scale.cuda()

    def forward(self, in0, in1, retPerLayer=False):
        in0_sc = (in0 - self.shift.expand_as(in0))/self.scale.expand_as(in0)
        in1_sc = (in1 - self.shift.expand_as(in0))/self.scale.expand_as(in0)

        outs0 = self.net.forward(in0_sc)
        outs1 = self.net.forward(in1_sc)

        if(retPerLayer):
            all_scores = []
        for (kk,out0) in enumerate(outs0):
            cur_score = (1.-util.cos_sim(outs0[kk],outs1[kk]))
            if(kk==0):
                val = 1.*cur_score
            else:
                # val = val + self.lambda_feat_layers[kk]*cur_score
                val = val + cur_score
            if(retPerLayer):
                all_scores+=[cur_score]

        if(retPerLayer):
            return (val, all_scores)
        else:
            return val

# Learned perceptual metric
class PNetLin(nn.Module):
    def __init__(self, pnet_type='vgg', pnet_rand=False, pnet_tune=False, use_dropout=True, use_gpu=True, spatial=False, version='0.1'):
        super(PNetLin, self).__init__()

        self.use_gpu = use_gpu
        self.pnet_type = pnet_type
        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
        self.spatial = spatial
        self.version = version

        if(self.pnet_type in ['vgg','vgg16']):
            net_type = pn.vgg16
            self.chns = [64,128,256,512,512]
        elif(self.pnet_type=='alex'):
            net_type = pn.alexnet
            self.chns = [64,192,384,256,256]
        elif(self.pnet_type=='squeeze'):
            net_type = pn.squeezenet
            self.chns = [64,128,256,384,384,512,512]

        if(self.pnet_tune):
            self.net = net_type(pretrained=not self.pnet_rand,requires_grad=True)
        else:
            self.net = [net_type(pretrained=not self.pnet_rand,requires_grad=False),]

        self.lin0 = NetLinLayer(self.chns[0],use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1],use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2],use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3],use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4],use_dropout=use_dropout)
        self.lins = [self.lin0,self.lin1,self.lin2,self.lin3,self.lin4]
        if(self.pnet_type=='squeeze'): # 7 layers for squeezenet
            self.lin5 = NetLinLayer(self.chns[5],use_dropout=use_dropout)
            self.lin6 = NetLinLayer(self.chns[6],use_dropout=use_dropout)
            self.lins+=[self.lin5,self.lin6]

        self.shift = torch.autograd.Variable(torch.Tensor([-.030, -.088, -.188]).view(1,3,1,1))
        self.scale = torch.autograd.Variable(torch.Tensor([.458, .448, .450]).view(1,3,1,1))

        if(use_gpu):
            if(self.pnet_tune):
                self.net.cuda()
            else:
                self.net[0].cuda()
            self.shift = self.shift.cuda()
            self.scale = self.scale.cuda()
            self.lin0.cuda()
            self.lin1.cuda()
            self.lin2.cuda()
            self.lin3.cuda()
            self.lin4.cuda()
            if(self.pnet_type=='squeeze'):
                self.lin5.cuda()
                self.lin6.cuda()

    def forward(self, in0, in1):
        in0_sc = (in0 - self.shift.expand_as(in0))/self.scale.expand_as(in0)
        in1_sc = (in1 - self.shift.expand_as(in0))/self.scale.expand_as(in0)

        if(self.version=='0.0'):
            # v0.0 - original release had a bug, where input was not scaled
            in0_input = in0
            in1_input = in1
        else:
            # v0.1
            in0_input = in0_sc
            in1_input = in1_sc

        if(self.pnet_tune):
            outs0 = self.net.forward(in0_input)
            outs1 = self.net.forward(in1_input)
        else:
            outs0 = self.net[0].forward(in0_input)
            outs1 = self.net[0].forward(in1_input)

        feats0 = {}
        feats1 = {}
        diffs = [0]*len(outs0)

        for (kk,out0) in enumerate(outs0):
            feats0[kk] = util.normalize_tensor(outs0[kk])
            feats1[kk] = util.normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk]-feats1[kk])**2

        if self.spatial:
            lin_models = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
            if(self.pnet_type=='squeeze'):
                lin_models.extend([self.lin5, self.lin6])
            res = [lin_models[kk].model(diffs[kk]) for kk in range(len(diffs))]
            return res
			
        val = torch.mean(torch.mean(self.lin0.model(diffs[0]),dim=3),dim=2)
        val = val + torch.mean(torch.mean(self.lin1.model(diffs[1]),dim=3),dim=2)
        val = val + torch.mean(torch.mean(self.lin2.model(diffs[2]),dim=3),dim=2)
        val = val + torch.mean(torch.mean(self.lin3.model(diffs[3]),dim=3),dim=2)
        val = val + torch.mean(torch.mean(self.lin4.model(diffs[4]),dim=3),dim=2)
        if(self.pnet_type=='squeeze'):
            val = val + torch.mean(torch.mean(self.lin5.model(diffs[5]),dim=3),dim=2)
            val = val + torch.mean(torch.mean(self.lin6.model(diffs[6]),dim=3),dim=2)

        val = val.view(val.size()[0],val.size()[1],1,1)

        return val

class Dist2LogitLayer(nn.Module):
    ''' takes 2 distances, puts through fc layers, spits out value between [0,1] (if use_sigmoid is True) '''
    def __init__(self, chn_mid=32,use_sigmoid=True):
        super(Dist2LogitLayer, self).__init__()
        layers = [nn.Conv2d(5, chn_mid, 1, stride=1, padding=0, bias=True),]
        layers += [nn.LeakyReLU(0.2,True),]
        layers += [nn.Conv2d(chn_mid, chn_mid, 1, stride=1, padding=0, bias=True),]
        layers += [nn.LeakyReLU(0.2,True),]
        layers += [nn.Conv2d(chn_mid, 1, 1, stride=1, padding=0, bias=True),]
        if(use_sigmoid):
            layers += [nn.Sigmoid(),]
        self.model = nn.Sequential(*layers)

    def forward(self,d0,d1,eps=0.1):
        return self.model.forward(torch.cat((d0,d1,d0-d1,d0/(d1+eps),d1/(d0+eps)),dim=1))

class BCERankingLoss(nn.Module):
    def __init__(self, use_gpu=True, chn_mid=32):
        super(BCERankingLoss, self).__init__()
        self.use_gpu = use_gpu
        self.net = Dist2LogitLayer(chn_mid=chn_mid)
        self.parameters = list(self.net.parameters())
        self.loss = torch.nn.BCELoss()
        self.model = nn.Sequential(*[self.net])

        if(self.use_gpu):
            self.net.cuda()

    def forward(self, d0, d1, judge):
        per = (judge+1.)/2.
        if(self.use_gpu):
            per = per.cuda()
        self.logit = self.net.forward(d0,d1)
        return self.loss(self.logit, per)

class NetLinLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = [nn.Dropout(),] if(use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),]
        self.model = nn.Sequential(*layers)


# L2, DSSIM metrics
class FakeNet(nn.Module):
    def __init__(self, use_gpu=True, colorspace='Lab'):
        super(FakeNet, self).__init__()
        self.use_gpu = use_gpu
        self.colorspace=colorspace

class L2(FakeNet):

    def forward(self, in0, in1):
        assert(in0.size()[0]==1) # currently only supports batchSize 1

        if(self.colorspace=='RGB'):
            (N,C,X,Y) = in0.size()
            value = torch.mean(torch.mean(torch.mean((in0-in1)**2,dim=1).view(N,1,X,Y),dim=2).view(N,1,1,Y),dim=3).view(N)
            return value
        elif(self.colorspace=='Lab'):
            value = util.l2(util.tensor2np(util.tensor2tensorlab(in0.data,to_norm=False)), 
                util.tensor2np(util.tensor2tensorlab(in1.data,to_norm=False)), range=100.).astype('float')
            ret_var = Variable( torch.Tensor((value,) ) )
            if(self.use_gpu):
                ret_var = ret_var.cuda()
            return ret_var

class DSSIM(FakeNet):

    def forward(self, in0, in1):
        assert(in0.size()[0]==1) # currently only supports batchSize 1

        if(self.colorspace=='RGB'):
            value = util.dssim(1.*util.tensor2im(in0.data), 1.*util.tensor2im(in1.data), range=255.).astype('float')
        elif(self.colorspace=='Lab'):
            value = util.dssim(util.tensor2np(util.tensor2tensorlab(in0.data,to_norm=False)), 
                util.tensor2np(util.tensor2tensorlab(in1.data,to_norm=False)), range=100.).astype('float')
        ret_var = Variable( torch.Tensor((value,) ) )
        if(self.use_gpu):
            ret_var = ret_var.cuda()
        return ret_var

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Network',net)
    print('Total number of parameters: %d' % num_params)
