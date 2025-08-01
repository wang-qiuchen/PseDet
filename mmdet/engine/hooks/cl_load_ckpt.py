from mmengine.hooks import Hook
from mmdet.registry import HOOKS
from mmengine.config import Config
import torch
import torch.nn as nn


@HOOKS.register_module()
class CLLoadCheckpoint(Hook):

    def __init__(self, ori_num_class=None)->None:
        self.ori_num_class=ori_num_class # not used   

    def after_load_checkpoint(self, runner, checkpoint: dict) -> None:
        runner.model.module.init_weights()
        if runner.model.module.bbox_head._get_name() == 'GFLHead' or\
            runner.model.module.bbox_head._get_name() == 'GFLHeadLossSeparate' or\
            runner.model.module.bbox_head._get_name() == 'GFLHeadPseudoDynamic' or\
            runner.model.module.bbox_head._get_name() == 'GFLHeadPseudo' or\
            runner.model.module.bbox_head._get_name() == 'GFLHeadPseudoDynamicSelective':
            # new_num_class = runner.model.module.bbox_head.gfl_cls.out_channels
            ori_num_class = checkpoint['state_dict']['bbox_head.gfl_cls.bias'].shape[0]
            added_classConv_weight = runner.model.module.bbox_head.gfl_cls.weight[ori_num_class:,...].cpu().clone().detach()
            added_classConv_bias = runner.model.module.bbox_head.gfl_cls.bias[ori_num_class:,...].cpu().clone().detach()
            checkpoint['state_dict']['bbox_head.gfl_cls.weight'] = torch.cat(
                (checkpoint['state_dict']['bbox_head.gfl_cls.weight'], added_classConv_weight), dim=0)
            checkpoint['state_dict']['bbox_head.gfl_cls.bias'] = torch.cat(
                (checkpoint['state_dict']['bbox_head.gfl_cls.bias'], added_classConv_bias), dim=0)
        elif runner.model.module.bbox_head._get_name() == 'ATSSHeadIncrement' or\
             runner.model.module.bbox_head._get_name() == 'ATSSHead' or\
             runner.model.module.bbox_head._get_name() == 'ATSSERDHeadIncrement':
            ori_num_class = checkpoint['state_dict']['bbox_head.atss_cls.bias'].shape[0]
            added_classConv_weight = runner.model.module.bbox_head.atss_cls.weight[ori_num_class:,...].cpu().clone().detach()
            added_classConv_bias = runner.model.module.bbox_head.atss_cls.bias[ori_num_class:,...].cpu().clone().detach()
            checkpoint['state_dict']['bbox_head.atss_cls.weight'] = torch.cat(
                (checkpoint['state_dict']['bbox_head.atss_cls.weight'], added_classConv_weight), dim=0)
            checkpoint['state_dict']['bbox_head.atss_cls.bias'] = torch.cat(
                (checkpoint['state_dict']['bbox_head.atss_cls.bias'], added_classConv_bias), dim=0)
        elif runner.model.module.bbox_head._get_name() == 'GFLHeadLossSeparateLoRA' : #lora
            # new_num_class = runner.model.module.bbox_head.gfl_cls.out_channels
            ori_num_class = checkpoint['state_dict']['bbox_head.gfl_cls.bias'].shape[0]
            
            added_classConv_weight = runner.model.module.bbox_head.gfl_cls.conv.weight[ori_num_class:,...].cpu().clone().detach()
            added_classConv_bias = runner.model.module.bbox_head.gfl_cls.conv.bias[ori_num_class:,...].cpu().clone().detach()
            
            checkpoint['state_dict']['bbox_head.gfl_cls.conv.weight'] = torch.cat(
                (checkpoint['state_dict']['bbox_head.gfl_cls.weight'], added_classConv_weight), dim=0)
            checkpoint['state_dict']['bbox_head.gfl_cls.conv.bias'] = torch.cat(
                (checkpoint['state_dict']['bbox_head.gfl_cls.bias'], added_classConv_bias), dim=0)
            
            checkpoint['state_dict'].pop('bbox_head.gfl_cls.weight')
            checkpoint['state_dict'].pop('bbox_head.gfl_cls.bias')
            
            ckpt_copy = checkpoint['state_dict'].copy()
            for k in checkpoint['state_dict'].keys():
                if 'cls_convs' in k and '.conv.' in k:
                    new_k = k.replace('.conv.','.conv.conv.')
                    ckpt_copy[new_k] = ckpt_copy[k]
                    ckpt_copy.pop(k)
            checkpoint['state_dict'] = ckpt_copy         
