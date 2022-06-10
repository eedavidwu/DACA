# Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import torch
import os
from torch.autograd import Variable
from torch.nn import init
import torch.nn as nn

import scipy.io as sio
import random
import functools
class FL_En_Module(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,stride,padding,activation=None):
        super(FL_En_Module, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,padding=padding)
        self.BN = nn.BatchNorm2d(out_channels)
        if activation=='sigmoid':
            self.activate_func=nn.Sigmoid()
        elif activation=='prelu':
            self.activate_func=nn.PReLU()
        elif activation==None:
            self.activate_func=None            

    def forward(self, inputs):
        out_conv1=self.conv1(inputs)
        out_bn=self.BN(out_conv1)
        if self.activate_func != None:
            out=self.activate_func(out_bn)
        else:
            out=out_bn
        return out

class FL_De_Module(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,stride,padding,out_padding,activation=None):
        super(FL_De_Module, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,padding=padding,output_padding=out_padding)
        self.GDN = nn.BatchNorm2d(out_channels)
        if activation=='sigmoid':
            self.activate_func=nn.Sigmoid()
        elif activation=='prelu':
            self.activate_func=nn.PReLU()
        elif activation==None:
            self.activate_func=None            

    def forward(self, inputs):
        out_deconv1=self.deconv1(inputs)
        out_bn=self.GDN(out_deconv1)
        if self.activate_func != None:
            out=self.activate_func(out_bn)
        else:
            out=out_bn
        return out

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out
# Encoder network
class Encoder(nn.Module):
    def __init__(self, input_nc, ngf=64, max_ngf=512, C_channel=16, n_blocks=2, n_downsampling=2, norm_layer=nn.BatchNorm2d, padding_type="reflect"):
        """Construct a Resnet-based encoder
        Parameters:
            input_nc (int)      -- the number of channels in input images
            ngf (int)           -- the number of filters in the first conv layer
            max_ngf (int)       -- the maximum number of filters
            C_channel (int)     -- the number of channels of the output
            n_blocks (int)      -- the number of ResNet blocks
            n_downsampling      -- number of downsampling layers
            norm_layer          -- normalization layer
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_downsampling>=0)
        assert(n_blocks>=0)
        super(Encoder, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        activation = nn.ReLU(True)
        model_pre = [nn.ReflectionPad2d((5-1)//2),
                 nn.Conv2d(3, 64, kernel_size=5, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 activation]
        self.model_pre = nn.Sequential(*model_pre)
        self.FL_down_sample_2=FL_En_Module(64, 128, 3, 2, 1,'prelu')
        self.FL_down_sample_3=FL_En_Module(128, 256, 3, 2, 1,'prelu')
        self.Res_4=ResnetBlock(256, padding_type=padding_type, norm_layer=norm_layer, use_dropout=False, use_bias=use_bias)
        self.Res_5=ResnetBlock(256, padding_type=padding_type, norm_layer=norm_layer, use_dropout=False, use_bias=use_bias)
        self.projection = nn.Conv2d(256, C_channel, kernel_size=3, padding=1, stride=1, bias=use_bias)

    def forward(self, input, H=None):
        #pre:
        out_pre= self.model_pre(input)
        #down:
        out_down_2=self.FL_down_sample_2(out_pre)
        out_down_3=self.FL_down_sample_3(out_down_2)
        #Res:
        out_res_4=self.Res_4(out_down_3)
        out_res_5=self.Res_5(out_res_4)
        out=self.projection(out_res_5)
        return  out

class Decoder(nn.Module):
    def __init__(self, output_nc, ngf=64, max_ngf=512, C_channel=16, n_blocks=2, n_downsampling=2, norm_layer=nn.BatchNorm2d, padding_type="reflect"):
        """Construct a Resnet-based generator

        Parameters:
            output_nc (int)     -- the number of channels for the output image
            ngf (int)           -- the number of filters in the first conv layer
            max_ngf (int)       -- the maximum number of filters
            C_channel (int)     -- the number of channels of the input
            n_blocks (int)      -- the number of ResNet blocks
            n_downsampling      -- number of downsampling layers
            norm_layer          -- normalization layer
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks>=0)
        assert(n_downsampling>=0)

        super(Decoder, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        activation = nn.ReLU(True)
        self.pre=nn.Conv2d(C_channel,256,kernel_size=3, padding=1 ,stride=1, bias=use_bias)
        self.Res_1=ResnetBlock(256, padding_type=padding_type, norm_layer=norm_layer, use_dropout=False, use_bias=use_bias)
        self.Res_2=ResnetBlock(256, padding_type=padding_type, norm_layer=norm_layer, use_dropout=False, use_bias=use_bias)
        self.Conv3 = FL_De_Module(256, 128, 3, stride=2,padding=1,out_padding=1,activation='prelu')
        self.Conv4 = FL_De_Module(128, 64, 3, stride=2,padding=1,out_padding=1,activation='prelu')
        last_layer=[nn.ReflectionPad2d(2), nn.Conv2d(64, 3, kernel_size=5, padding=0),nn.Sigmoid()]
        self.last_layer = nn.Sequential(*last_layer)

    def forward(self, input):
        out_pre=self.pre(input)
        out_res_1=self.Res_1(out_pre)
        out_res_2=self.Res_1(out_res_1)
        out_conv3=self.Conv3(out_res_2)
        out_conv4=self.Conv4(out_conv3)
        out=self.last_layer(out_conv4)
        return (2*out-1)

class Subnet(nn.Module):
    def __init__(self, dim, dim_out, dim_in, norm_layer):
        super(Subnet, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.conv_block = self.build_conv_block(dim, dim_out, dim_in, norm_layer, use_bias)

    def build_conv_block(self, dim, dim_out, dim_in, norm_layer, use_bias):
        conv_block = []
        conv_block += [nn.Conv2d(dim, dim_in, kernel_size=5, padding=2, bias=use_bias), norm_layer(dim_in), nn.ReLU(True)]
        conv_block += [nn.Conv2d(dim_in, dim_in, kernel_size=5, padding=2, bias=use_bias), norm_layer(dim_in), nn.ReLU(True)]
        conv_block += [nn.Conv2d(dim_in, dim_out, kernel_size=5, padding=2, bias=use_bias)]
        return nn.Sequential(*conv_block)
    def forward(self, x): 
        return self.conv_block(x) 

class JSCCOFDMModel(nn.Module):
    def __init__(self,args):
        super(JSCCOFDMModel, self).__init__()
        self.pilot=(self.zcsequence(3,64)).repeat(1,1)
        #self.netEQ = networks.define_S(dim=6, dim_out=2, dim_in = 32,
        #                            norm=opt.norm_EG, init_type='kaiming', init_gain=0.02, gpu_ids=self.gpu_ids)
        Bn = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        self.netP=Subnet(dim=(args.tcn+2), dim_out=args.tcn, dim_in =64, norm_layer=Bn)
        # define networks (both generator and discriminator)
        self.netE =Encoder(input_nc=3, ngf=64, max_ngf=256, C_channel=args.tcn, n_blocks=2, n_downsampling=2, norm_layer=Bn, padding_type="reflect")

        self.netG =Decoder(output_nc=3, ngf=64, max_ngf=256, C_channel=args.tcn, n_blocks=2, n_downsampling=2, norm_layer=Bn, padding_type="reflect")
    def zcsequence(self,u, seq_length, q=0):
        for el in [u,seq_length,q]:
            if not float(el).is_integer():
                raise ValueError('{} is not an integer'.format(el))
        if u<=0:
            raise ValueError('u is not stricly positive')
        if u>=seq_length:
            raise ValueError('u is not stricly smaller than seq_length')
        if np.gcd(u,seq_length)!=1:
            raise ValueError('the greatest common denominator of u and seq_length is not 1')

        cf = seq_length%2
        n = np.arange(seq_length)
        zcseq = np.exp( -1j * np.pi * u * n * (n+cf+2.*q) / seq_length)
        zcseq=torch.from_numpy(zcseq)
        real_part=torch.real(zcseq).float()
        img_part=torch.imag(zcseq).float()
        zcseq=torch.complex(real_part,img_part)
        return zcseq

    def compute_h_broadcast(self,batch_size,subchannel_num):
        h_stddev=torch.full((batch_size,subchannel_num),1/(np.sqrt(2))).float()
        h_mean=torch.zeros_like(h_stddev).float()
        h_real=Variable(torch.normal(mean=h_mean,std=h_stddev)).float()
        h_img=Variable(torch.normal(mean=h_mean,std=h_stddev)).float()
        h=torch.complex(h_real,h_img)
        return h

    def power_normalize(self,feature):
        in_shape=feature.shape
        batch_size=in_shape[0]
        z_in=feature.reshape(batch_size,-1)
        sig_pwr=torch.square(torch.abs(z_in))
        ave_sig_pwr=sig_pwr.mean(dim=1).unsqueeze(dim=1)
        z_in_norm=z_in/(torch.sqrt(ave_sig_pwr))
        inputs_in_norm=z_in_norm.reshape(in_shape)
        return inputs_in_norm
            
    def transmit(self,feature,snr,h):
        feature_shape=feature.shape
        #prepare h: [batch,S,M)
        h_broadcast=h.unsqueeze(dim=1).repeat(1,feature_shape[1],1)
        #prepare W: [batch,S,M)
        noise_stddev=(torch.sqrt(10**(-snr/10))/torch.sqrt(torch.tensor(2))).reshape(-1,1,1)
        noise_stddev_board=noise_stddev.repeat(1,feature_shape[1],feature_shape[2]).float()
        mean=torch.zeros_like(noise_stddev_board).float()
        noise_real=Variable(torch.normal(mean=mean,std=noise_stddev_board).cuda()).float()
        noise_img=Variable(torch.normal(mean=mean,std=noise_stddev_board).cuda()).float()
        w=torch.complex(noise_real,noise_img)
        #compute channel_out
        channel_out=h_broadcast*feature+w
        return channel_out

    def MMSE_est(self,x,y,snr):
        #est=R_h(X^hXR_h+var_wI)^-1X^hy
        #x:(b,1,M) y:C(b,1,M)
        #h_std:(b,1),snr:(b,1)
        #h_est:(b,1,M)
        input_shape=x.shape
        batch_size=input_shape[0]
        subchannel_num=input_shape[2]
        w_noise_varian_r=((10**(-snr/10))).repeat(1,subchannel_num).float().cuda()
        #w_noise_varian_r=torch.zeros_like(w_noise_varian_r)
        w_noise_varian_i=torch.zeros_like(w_noise_varian_r)
        w_noise_varian=torch.complex(w_noise_varian_r,w_noise_varian_i)
        h_stddev=torch.full((batch_size,subchannel_num),1).float().cuda()
        #R_h
        R_h_r=torch.diag_embed(h_stddev)
        R_h_i=torch.zeros_like(R_h_r)
        R_h=torch.complex(R_h_r,R_h_i)
        y=y.squeeze().unsqueeze(2)
        x=torch.diag_embed(x.squeeze())
        mid=torch.bmm(torch.bmm(self.conj_transpose(x),x),R_h)+torch.diag_embed(w_noise_varian)
        h_est=torch.bmm(torch.bmm(torch.bmm(R_h,torch.inverse(mid)),self.conj_transpose(x)),y)
        return h_est
    def conj_transpose(self,x):
        return torch.conj(x).permute(0,2,1)   
    def forward(self,x,input_snr):
        batch_size=x.shape[0]
        Y_p=self.pilot.repeat(batch_size,1,1)
        Y_p_norm=Y_p.cuda()
        x_head_ave=torch.zeros_like(x).cuda()    
        snr=np.full(batch_size,input_snr)
        channel_snr=torch.from_numpy(snr).float().cuda().view(-1,1).cuda()
        x_head_ave=torch.zeros_like(x).cuda()    

        for transmit_times in range (5):
            #prepare H:
            #h:C[batch,M]; h_attention:R[batch,2M]
            h=self.compute_h_broadcast(batch_size,64).cuda()
            #h_norm=torch.abs(h)
            #transmit pilot to estimate h
            Y_p_norm_head=self.transmit(Y_p_norm,channel_snr,h)
            h_est_mmse=self.MMSE_est(Y_p_norm,Y_p_norm_head,channel_snr).squeeze()
            #h_est_mmse_norm=torch.abs(h_est_mmse)
            h_est=h_est_mmse       
            latent = self.netE(x)

            encoded_shape=latent.shape
            H = h_est.view(-1, latent.shape[2], latent.shape[3])
            H_r=torch.real(H).unsqueeze(dim=1)
            H_im=torch.imag(H).unsqueeze(dim=1)
            H_all=torch.cat((H_r,H_im),dim=1)
            weights = self.netP(torch.cat((H_all, latent), 1))
            latent = latent*weights

            ####Power constraint for each sending:
            z=latent.view(batch_size,-1)
            complex_list=torch.split(z,(z.shape[1]//2),dim=1)
            encoded_out_complex=torch.complex(complex_list[0],complex_list[1])
            Y=encoded_out_complex.view(batch_size,-1,64)          
            Y_norm=self.power_normalize(Y)
            Y_norm_head=self.transmit(Y_norm,channel_snr,h)

            h_broadcast_est=h_est.unsqueeze(dim=1).repeat(1,Y_norm_head.shape[1],1)
            compensation_factor=(torch.conj(h_broadcast_est)/(torch.square(torch.abs(h_broadcast_est))))
            Y_norm_head=compensation_factor*Y_norm_head
            Y_norm_head=Y_norm_head.view(batch_size,-1)
            Y_norm_head_real=torch.cat((torch.real(Y_norm_head),torch.imag(Y_norm_head)),dim=1).view(encoded_shape).float()
            decoder_input=Y_norm_head_real

            x_head = self.netG(decoder_input)
            x_head_ave=x_head_ave+x_head
        x_head_ave=x_head_ave/5
        return x_head_ave
            
    '''
    def channel_estimation(self, out_pilot, noise_pwr):
        return channel.LMMSE_channel_est(self.channel.pilot, out_pilot, self.opt.M*noise_pwr)

    def equalization(self, H_est, out_sig, noise_pwr):
        return channel.MMSE_equalization(H_est, out_sig, self.opt.M*noise_pwr)
    '''
        

