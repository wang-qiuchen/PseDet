# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch.nn as nn
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner

from mmdet.registry import HOOKS

import copy


@HOOKS.register_module()
class SelectiveUpdateHook(Hook):
    """Mean Teacher Hook.

    Mean Teacher is an efficient semi-supervised learning method in
    `Mean Teacher <https://arxiv.org/abs/1703.01780>`_.
    This method requires two models with exactly the same structure,
    as the student model and the teacher model, respectively.
    The student model updates the parameters through gradient descent,
    and the teacher model updates the parameters through
    exponential moving average of the student model.
    Compared with the student model, the teacher model
    is smoother and accumulates more knowledge.

    Args:
        momentum (float): The momentum used for updating teacher's parameter.
            Teacher's parameter are updated with the formula:
           `teacher = (1-momentum) * teacher + momentum * student`.
            Defaults to 0.001.
        interval (int): Update teacher's parameter every interval iteration.
            Defaults to 1.
        skip_buffers (bool): Whether to skip the model buffers, such as
            batchnorm running stats (running_mean, running_var), it does not
            perform the ema operation. Default to True.
    """

    def __init__(self,
                 momentum: float = 0.001,
                 interval: int = 1,
                 skip_buffer=True,
                 select_keep=False,
                 k_percent=0.1,
                 cls_head_only=False,
                 branch_only=False,
                 branch_and_fpn_only = False,
                 retype = 'grad') -> None:
        assert 0 < momentum < 1
        self.momentum = momentum
        self.interval = interval
        self.skip_buffers = skip_buffer
        self.retype = retype
        self.select_keep = select_keep
        self.k_percent = k_percent
        self.cls_head_only = cls_head_only
        self.branch_only = branch_only
        self.branch_and_fpn_only = branch_and_fpn_only

    # def before_train(self, runner: Runner) -> None:
    #     """To check that teacher model and student model exist."""
    #     model = runner.model
    #     if is_model_wrapper(model):
    #         model = model.module
        # assert hasattr(model, 'teacher')
        # assert hasattr(model, 'student')
        # # only do it at initial stage
        # if runner.iter == 0:
        #     self.momentum_update(model, 1)
        # for name, parameters in model.named_parameters():
        #     if parameters.requires_grad:
        #         parameters.retain_grad()
        #     else:
        #         print(name)

    def after_train_iter(self,
                         runner: Runner,
                         batch_idx: int,
                         data_batch: Optional[dict] = None,
                         outputs: Optional[dict] = None) -> None:
        """Update teacher's parameter every self.interval iterations."""
        # if (runner.iter + 1) % self.interval != 0:
        #     return
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        
        import pdb;pdb.set_trace()

        if runner.iter==0:
            self.former = copy.deepcopy(model.state_dict())
        selected_params=[]
        selected_layers=[]
        grad_sum_per_layer = {}  
        for name, param in model.named_parameters():
            if param.grad is not None:  
                layer_name = name
                assert(layer_name not in grad_sum_per_layer)
                if self.retype == 'diff':
                    grad_sum_per_layer[layer_name] = (param.data - self.former[name]).abs().sum().item()
                elif self.retype=='grad':
                    grad_sum_per_layer[layer_name] = param.grad.abs().sum().item()
                    if grad_sum_per_layer[layer_name] != 0:
                        import pdb;pdb.set_trace()
        sorted_layers = sorted(grad_sum_per_layer.keys(), key=lambda x: grad_sum_per_layer[x], reverse=True)
        k_percent = self.k_percent

        if self.cls_head_only:
            filtered_list = [item for item in sorted_layers if 'cls' in item]
            sorted_layers = filtered_list
        
        if self.branch_only:
            # import pdb;pdb.set_trace()
            filtered_list = [item for item in sorted_layers if 'bbox_head' in item]
            sorted_layers = filtered_list
        
        if self.branch_and_fpn_only:
            filtered_list = [item for item in sorted_layers if 'bbox_head' in item or 'neck' in item]
            sorted_layers = filtered_list

        num_layers_to_select = int(len(sorted_layers) * k_percent)
        for layer_name in sorted_layers[:num_layers_to_select]:
            already=False
            for selected in selected_layers:
                if layer_name==selected:
                    already=True
                    break
            if already==False:
                selected_params.extend([param for name, param in model.named_parameters() if layer_name in name])
                selected_layers.append(layer_name)
        

        
        for name,para in  model.named_parameters():
            if self.select_keep:
                if name in selected_layers:
                    para.data=self.former[name]
            else:
                if name not in selected_layers:
                    para.data=self.former[name]


        self.former = copy.deepcopy(model.state_dict())
        # self.momentum_update(model, self.momentum)

    def momentum_update(self, model: nn.Module, momentum: float) -> None:
        """Compute the moving average of the parameters using exponential
        moving average."""
        if self.skip_buffers:
            for (src_name, src_parm), (dst_name, dst_parm) in zip(
                    model.student.named_parameters(),
                    model.teacher.named_parameters()):
                dst_parm.data.mul_(1 - momentum).add_(
                    src_parm.data, alpha=momentum)
        else:
            for (src_parm,dst_parm) in zip(model.student.state_dict().values(),
                                  model.teacher.state_dict().values()):
                # exclude num_tracking
                if dst_parm.dtype.is_floating_point:
                    dst_parm.data.mul_(1 - momentum).add_(src_parm.data, alpha=momentum)
