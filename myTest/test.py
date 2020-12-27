# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 19:33:09 2020

@author: Administrator
"""

import os
import sys
import argparse
import yaml
import numpy as np
from easydict import EasyDict
import torch
import torch.nn as nn
from torch.autograd import Variable
import cv2
#import matplotlib
import matplotlib.pyplot as plt

sys.path.append('../model')
sys.path.append('../data')
import model_mobilenetv2_seg_small as modellib
from data_aug import Normalize_Img

def resize_padding(image, dstshape, padValue=0):
  height, width, _ = image.shape
  ratio = float(width)/height # ratio = (width:height)
  dst_width = int(min(dstshape[1]*ratio, dstshape[0]))
  dst_height = int(min(dstshape[0]/ratio, dstshape[1]))
  origin = [int((dstshape[1] - dst_height)/2), int((dstshape[0] - dst_width)/2)]
  if len(image.shape)==3:
    image_resize = cv2.resize(image, (dst_width, dst_height))
    newimage = np.zeros(shape = (dstshape[1], dstshape[0], image.shape[2]), dtype = np.uint8) + padValue
    newimage[origin[0]:origin[0]+dst_height, origin[1]:origin[1]+dst_width, :] = image_resize
    bbx = [origin[1], origin[0], origin[1]+dst_width, origin[0]+dst_height] # x1,y1,x2,y2
  else:
    image_resize = cv2.resize(image, (dst_width, dst_height),  interpolation = cv2.INTER_NEAREST)
    newimage = np.zeros(shape = (dstshape[1], dstshape[0]), dtype = np.uint8)
    newimage[origin[0]:origin[0]+height, origin[1]:origin[1]+width] = image_resize
    bbx = [origin[1], origin[0], origin[1]+dst_width, origin[0]+dst_height] # x1,y1,x2,y2
  return newimage, bbx
  

def generate_input(exp_args, inputs, prior=None):
  inputs_norm = Normalize_Img(inputs, scale=exp_args.img_scale, mean=exp_args.img_mean, val=exp_args.img_val)
    
  if exp_args.video == True:
    if prior is None:
      prior = np.zeros((exp_args.input_height, exp_args.input_width, 1))
      inputs_norm = np.c_[inputs_norm, prior]
    else:
      prior = prior.reshape(exp_args.input_height, exp_args.input_width, 1)
      inputs_norm = np.c_[inputs_norm, prior]
       
  inputs = np.transpose(inputs_norm, (2, 0, 1))
  return np.array(inputs, dtype=np.float32)

def pred_single(model, exp_args, img_ori, prior=None):
  model.eval()
  softmax = nn.Softmax(dim=1)
    
  in_shape = img_ori.shape
  img, bbx = resize_padding(img_ori, [exp_args.input_height, exp_args.input_width], padValue=exp_args.padding_color)
    
  in_ = generate_input(exp_args, img, prior)
  in_ = in_[np.newaxis, :, :, :]
    
  if exp_args.addEdge == True:
    output_mask, output_edge = model(Variable(torch.from_numpy(in_)).cuda())
  else:
    output_mask = model(Variable(torch.from_numpy(in_)).cuda())
  prob = softmax(output_mask)
  pred = prob.data.cpu().numpy()
    
  predimg = pred[0].transpose((1,2,0))[:,:,1]
  out = predimg[bbx[1]:bbx[3], bbx[0]:bbx[2]]
  out = cv2.resize(out, (in_shape[1], in_shape[0]))
  return out, predimg


def get_args(args):
  with open(args.config_path, 'r') as f:
    cont = f.read()
    f.close()
  config = yaml.safe_load(cont)
  print(config)
  exp_args = EasyDict()
  exp_args.data_root = args.data_root
  exp_args.model_root = args.model_root
  exp_args.input_height = config['input_height']
  exp_args.input_width = config['input_width']
  exp_args.video = config['video']
  exp_args.prior_prob = config['prior_prob']
  exp_args.addEdge = config['addEdge']
  exp_args.stability = config['stability']
  exp_args.padding_color = config['padding_color']
  exp_args.img_scale = config['img_scale']
  exp_args.img_mean = config['img_mean']
  exp_args.img_val = config['img_val']
  exp_args.useUpsample = config['useUpsample']
  exp_args.useDeconvGroup = config['useDeconvGroup']
  print(exp_args)
  return exp_args

def test_main(args):
  exp_args = get_args(args)
  
  net = modellib.MobileNetV2(n_class=2,
                             useUpsample=exp_args.useUpsample,
                             useDeconvGroup=exp_args.useDeconvGroup,
                             addEdge=exp_args.addEdge,
                             channelRatio=1.0,
                             minChannel=16,
                             weightInit=True,
                             video=exp_args.video).cuda()
  #print(net)
  best_model_file = '/'.join((args.model_root, 'model_best.pth.tar'))
  if( os.path.isfile(best_model_file)):
    print('load model: ', best_model_file)
    model = torch.load(best_model_file)
    net.load_state_dict(model['state_dict'])
    print('minLoss: ', model['minLoss'], 'epoch: ', model['epoch'])
  else:
    print('model file does not exist')
    return
  
  img_ori = cv2.imread(exp_args.data_root + 'EG1800/Images/00693.png')
  height, width, _ = img_ori.shape
  background = img_ori.copy()
  background = cv2.blur(background, (17, 17))
  alphargb, pred = pred_single(net, exp_args, img_ori, None)
  plt.imshow(alphargb)
  plt.show()
  alphargb = cv2.cvtColor(alphargb, cv2.COLOR_GRAY2BGR)  

  result = np.uint8(img_ori * alphargb + background * (1-alphargb))
  
  myImg = np.ones((height, width*2 + 20, 3)) * 255
  myImg[:, :width, :] = img_ori
  myImg[:, width+20:, :] = result
  plt.imshow(myImg[:,:,::-1]/255)
  plt.show()


if __name__ == '__main__':
  cur_path = os.path.split(os.path.realpath(__file__))[0]
  parser = argparse.ArgumentParser(description='Test code')
  parser.add_argument('--model_root',
                      default=cur_path + '/../modelroot/',
                      type=str, help='the model root')
  parser.add_argument('--config_path',
                      default=cur_path + '/../config/model_mobilenetv2_with_two_auxiliary_losses.yaml',
                      type=str, help='the config path of the model')
  parser.add_argument('--data_root',
                      default=cur_path + '/../data/',
                      type=str, help='the data root')
  args = parser.parse_args()
  test_main(args)

