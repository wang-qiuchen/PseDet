from mmengine.hooks import Hook
from mmdet.registry import HOOKS


@HOOKS.register_module()
class SdmInitWeightHook(Hook):

    def __init__(self, learnable_sdm, domain_name, domain_num, init_sdm_from_former)->None:
        self.learnable_sdm = learnable_sdm
        self.domain_name = domain_name
        self.domain_num = domain_num
        self.init_sdm_from_former = init_sdm_from_former
        self.sdm_index = ['D0','D1','D2','D3'].index(self.domain_name) if self.domain_num == 4 else 0

    def after_load_checkpoint(self, runner, checkpoint: dict) -> None:
        if self.domain_num == 4 and self.init_sdm_from_former:
            checkpoint['state_dict']['bbox_head.sdm'][self.sdm_index] = checkpoint['state_dict']['bbox_head.sdm'][self.sdm_index-1]


        # checkpoint['state_dict']['bbox_head.k_proj.%s.weight'] % sdm_index = checkpoint['state_dict']['bbox_head.k_proj.1.weight']
