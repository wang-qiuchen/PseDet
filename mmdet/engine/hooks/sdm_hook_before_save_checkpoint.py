from mmengine.hooks import Hook
from mmdet.registry import HOOKS


@HOOKS.register_module()
class SdmHookBeforeSaveCheckpoint(Hook):

    def __init__(self, learnable_sdm, domain_name, domain_num, init_sdm_from_former, init_sdmProj_from_former, init_sdmMlp_from_former)->None:
        self.learnable_sdm = learnable_sdm
        self.domain_name = domain_name
        self.domain_num = domain_num
        self.init_sdm_from_former = init_sdm_from_former
        self.init_sdmProj_from_former = init_sdmProj_from_former
        self.init_sdmMlp_from_former = init_sdmMlp_from_former
        self.sdm_index = ['D0','D1','D2','D3'].index(self.domain_name) if self.domain_num == 4 else 0

    def before_save_checkpoint(self, runner, checkpoint: dict) -> None:
        if self.domain_num == 4 and self.sdm_index < self.domain_num-1 and self.init_sdm_from_former:
            checkpoint['state_dict']['bbox_head.sdm'][self.sdm_index+1] = checkpoint['state_dict']['bbox_head.sdm'][self.sdm_index]
        if self.domain_num == 4 and self.sdm_index < self.domain_num-1 and self.init_sdmProj_from_former:
            q_key = 'bbox_head.q_proj.%s.weight' % self.sdm_index
            k_key = 'bbox_head.k_proj.%s.weight' % self.sdm_index
            v_key = 'bbox_head.v_proj.%s.weight' % self.sdm_index
            q_key_next = 'bbox_head.q_proj.%s.weight' % (self.sdm_index+1)
            k_key_next = 'bbox_head.k_proj.%s.weight' % (self.sdm_index+1)
            v_key_next = 'bbox_head.v_proj.%s.weight' % (self.sdm_index+1)
            checkpoint['state_dict'][q_key_next] = checkpoint['state_dict'][q_key]
            checkpoint['state_dict'][k_key_next] = checkpoint['state_dict'][k_key]
            checkpoint['state_dict'][v_key_next] = checkpoint['state_dict'][v_key]            
        if self.domain_num == 4 and self.sdm_index < self.domain_num-1 and self.init_sdmMlp_from_former:
            center_key = 'bbox_head.atss_centerness.%s.weight' % self.sdm_index
            center_key_next = 'bbox_head.atss_centerness.%s.weight' % (self.sdm_index+1)
            checkpoint['state_dict'][center_key_next] = checkpoint['state_dict'][center_key]
            center_key_bias = 'bbox_head.atss_centerness.%s.bias' % self.sdm_index
            center_key_bias_next = 'bbox_head.atss_centerness.%s.bias' % (self.sdm_index+1)
            checkpoint['state_dict'][center_key_bias_next] = checkpoint['state_dict'][center_key_bias]

            cls_key = 'bbox_head.atss_cls.%s.weight' % self.sdm_index
            cls_key_next = 'bbox_head.atss_cls.%s.weight' % (self.sdm_index+1)
            checkpoint['state_dict'][cls_key_next] = checkpoint['state_dict'][cls_key]
            cls_key_bias = 'bbox_head.atss_cls.%s.bias' % self.sdm_index
            cls_key_bias_next = 'bbox_head.atss_cls.%s.bias' % (self.sdm_index+1)
            checkpoint['state_dict'][cls_key_bias_next] = checkpoint['state_dict'][cls_key_bias]

            for index_scale in range(5):
                scale_key = 'bbox_head.scales.%s.%s.scale' % (self.sdm_index,index_scale)
                scale_key_next = 'bbox_head.scales.%s.%s.scale' % (self.sdm_index+1,index_scale)
                checkpoint['state_dict'][scale_key_next] = checkpoint['state_dict'][scale_key]

            
