# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 08:27:47 2020

https://github.com/sksq96/pytorch-summary

@author: Volvet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from collections import OrderedDict


def summary_string(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):
  if dtypes == None:
    dtypes = [torch.FloatTensor]*len(input_size)
    
  summary_str = ''
  format_str = '{:>20} {:>30} {:>15}'
  summary_dict = OrderedDict()

  def register_hook(module):
    def hook(module, input, output):
      class_name = str(module.__class__).split('.')[-1].split('\'')[0]
      module_idx = len(summary_dict)
      
      m_key = '%s-%i' % (class_name, module_idx+1)
      summary_dict[m_key] = OrderedDict()
      summary_dict[m_key]['input_shape'] = list(input[0].size())
      summary_dict[m_key]['input_shape'][0] = batch_size
      if isinstance(output, (list, tuple)):
        summary_dict[m_key]['output_shape'] = [
            [-1] + list(o.size())[1:] for o in output
          ]
      else:
        summary_dict[m_key]['output_shape'] = list(output.size())
        summary_dict[m_key]['output_shape'][0] = batch_size

      params = 0
      if hasattr(module, 'weight') and hasattr(module.weight, 'size'):
        params += torch.prod(torch.LongTensor(list(module.weight.size())))
        summary_dict[m_key]['trainable'] = module.weight.requires_grad
      if hasattr(module, 'bias') and hasattr(module.bias, 'size'):
        params += torch.prod(torch.LongTensor(list(module.bias.size())))
      summary_dict[m_key]['nb_params'] = params
      
    if( not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) ):
      hooks.append(module.register_forward_hook(hook))
  
  if isinstance(input_size, tuple):
    input_size = [input_size]
  x = [torch.rand(2, *in_size).type(dtype).to(device=device) for in_size, dtype in zip(input_size, dtypes)]
  hooks = []
  model.apply(register_hook)
  model(*x)
  
  for h in hooks:
    h.remove()

  summary_str += '--------------------------------------------------------------------------' + '\n'
  line_new = format_str.format('Layer (type)', 'Output Shape', 'Param #')
  summary_str += line_new + '\n'
  summary_str += '==========================================================================' + '\n'
  
  total_params = 0
  total_output = 0
  trainable_params = 0
  for layer in summary_dict:
    line_new = format_str.format(layer, 
                                 str(summary_dict[layer]['output_shape']),
                                 '{:,}'.format(summary_dict[layer]['nb_params']))
    total_params += summary_dict[layer]['nb_params']
    total_output += np.prod(summary_dict[layer]['output_shape'])
    if 'trainable' in summary_dict[layer]:
      if summary_dict[layer]['trainable']:
        trainable_params += summary_dict[layer]['nb_params']
    summary_str += line_new + '\n'
  total_input_size = abs(np.prod(sum(input_size, ())) * batch_size * 4. / (1024**2.))
  total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))
  total_params_size = abs(total_params * 4. / (1024 ** 2.))
  total_size = total_params_size + total_output_size + total_input_size
  
  summary_str += '=========================================================================' + '\n'
  summary_str += 'Total params: {0:,}'.format(total_params) + '\n'
  summary_str += 'Trainable params: {0:,}'.format(trainable_params) + '\n'
  summary_str += 'Non-tranable params: {0:,}'.format(total_params - trainable_params) + '\n'
  summary_str += '-------------------------------------------------------------------------' + '\n'

  summary_str += 'Input size(MB): %0.2f' % (total_input_size) + '\n'
  summary_str += 'Forward/backward pass size(MB): %0.2f' % (total_output_size) + '\n'
  summary_str += 'Param size(MB): %0.2f' % (total_params_size) + '\n'
  summary_str += 'Estimated total size(MB): %0.2f' % (total_size) + '\n'
  return summary_str, (total_params, trainable_params)
  

def summary(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):
  result, _ = summary_string(model, input_size, batch_size, device, dtypes)
  print(result)

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.conv2_drop = nn.Dropout2d()
    self.fc1 = nn.Linear(320, 50)
    self.fc2 = nn.Linear(50, 10)

  def forward(self, x):
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
    x = x.view(-1, 320)
    x = F.relu(self.fc1(x))
    x = F.dropout(x, training=self.training)
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)


if __name__ == '__main__':
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
  model = Net().to(device)
  summary(model, (1, 28, 28))
  
  #vgg = models.vgg16().to(device)
  #summary(vgg, (3, 224, 224))

