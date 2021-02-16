# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 13:03:43 2021

@author: Administrator
"""

import torch
import torch.onnx
import coremltools as ct

def export_to_onnx(args, exp_args):
  if args.model == 'PortraitNet':
    import model_mobilenetv2_seg_small as modellib
    netmodel = modellib.MobileNetV2(n_class=2,
                                    useUpsample=exp_args.useUpsample,
                                    useDeconvGroup=exp_args.useDeconvGroup,
                                    addEdge=exp_args.addEdge,
                                    channelRatio=1.0,
                                    minChannel=16,
                                    weightInit=True,
                                    video=exp_args.video).cuda()

    weights_path = '/'.join((exp_args.model_root, 'model_best.pth.tar'))
    onnx_path = '/'.join((exp_args.model_root, 'PortaitNet.onnx'))
    state_dict = torch.load(weights_path)
    netmodel.load_state_dict(state_dict['state_dict'])
    batch_size = 1 #args.batchsize
    channels = 3
    input_width = exp_args.input_width
    input_height = exp_args.input_height
    dummy_input = torch.randn(batch_size, channels, input_width, input_height, device='cuda')
    torch.onnx.export(netmodel, dummy_input, onnx_path, verbose=False)


def export_to_coreml(args, exp_args):
  if args.model == 'PortraitNet':
    import model_mobilenetv2_seg_small as modellib
    netmodel = modellib.MobileNetV2(n_class=2,
                                    useUpsample=exp_args.useUpsample,
                                    useDeconvGroup=exp_args.useDeconvGroup,
                                    addEdge=exp_args.addEdge,
                                    channelRatio=1.0,
                                    minChannel=16,
                                    weightInit=True,
                                    video=exp_args.video).cuda()
    #netmodel = netmodel.eval()
    weights_path = '/'.join((exp_args.model_root, 'model_best.pth.tar'))
    coreml_path = '/'.join((exp_args.model_root, 'PortaitNet.mlmodel'))
    state_dict = torch.load(weights_path)
    netmodel.load_state_dict(state_dict['state_dict'])
    batch_size = 1 #args.batchsize
    channels = 3
    input_width = exp_args.input_width
    input_height = exp_args.input_height
    dummy_input = torch.randn(batch_size, channels, input_width, input_height, device='cuda')
    netmodel = netmodel.eval()
    traced_model = torch.jit.trace(netmodel, dummy_input)
    mlmodel = ct.convert(traced_model,
                         inputs=[ct.TensorType(name="captured_input", shape=dummy_input.shape)])
    mlmodel.save(coreml_path) 


if __name__ == '__main__':
  pass