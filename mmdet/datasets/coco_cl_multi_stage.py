# Copyright (c) OpenMMLab. All rights reserved.
from tracemalloc import start
import copy
import os.path as osp
from rich import print
from typing import List, Union
from typing import List, Optional

from mmengine.fileio import get_local_path

from mmdet.registry import DATASETS
from .api_wrappers import COCO
from .base_det_dataset import BaseDetDataset
from mmdet.datasets import CocoDataset
import random
from mmengine import load
import torch
import numpy as np

@DATASETS.register_module()
class CocoDatasetCLMultiStage(CocoDataset):
    """Dataset for COCO of Continual Learning"""

    METAINFO = {
        'classes':
        ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
         'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
         'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
         'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
         'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
         'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
         'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
         'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
         'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
         'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
         'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
         'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
         'scissors', 'teddy bear', 'hair drier', 'toothbrush'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
         (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
         (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
         (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
         (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255),
         (199, 100, 0), (72, 0, 118), (255, 179, 240), (0, 125, 92),
         (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164),
         (92, 0, 73), (133, 129, 255), (78, 180, 255), (0, 228, 0),
         (174, 255, 243), (45, 89, 255), (134, 134, 103), (145, 148, 174),
         (255, 208, 186), (197, 226, 255), (171, 134, 1), (109, 63, 54),
         (207, 138, 255), (151, 0, 95), (9, 80, 61), (84, 105, 51),
         (74, 65, 105), (166, 196, 102), (208, 195, 210), (255, 109, 65),
         (0, 143, 149), (179, 0, 194), (209, 99, 106), (5, 121, 0),
         (227, 255, 205), (147, 186, 208), (153, 69, 1), (3, 95, 161),
         (163, 255, 0), (119, 0, 170), (0, 182, 199), (0, 165, 120),
         (183, 130, 88), (95, 32, 0), (130, 114, 135), (110, 129, 133),
         (166, 74, 118), (219, 142, 185), (79, 210, 114), (178, 90, 62),
         (65, 70, 15), (127, 167, 115), (59, 105, 106), (142, 108, 45),
         (196, 172, 0), (95, 54, 80), (128, 76, 255), (201, 57, 1),
         (246, 0, 122), (191, 162, 208)]
    }
    COCOAPI = COCO
    # ann_id is unique in coco dataset.
    ANN_ID_UNIQUE = False

    def __init__(self,
                 *args,
                 seg_map_suffix: str = '.png',
                 proposal_file: Optional[str] = None,
                 file_client_args: dict = None,
                 backend_args: dict = None,
                 pseudo_file=None,
                 score_thr = 0.,
                 seed = 123,
                 temporal = False,
                 **kwargs) -> None:
        self.seed = seed
        self.temporal = temporal
        if pseudo_file is not None and temporal:
            self.pseudo_ann_list = []
            for pseudo_file_name in pseudo_file:
                print('loading pseudo label from %s'%pseudo_file_name)
                ann = load(pseudo_file_name)
                self.pseudo_ann_list.append(ann)
            self.img_index = {ann['img_id']: i for i, ann in enumerate(self.pseudo_ann_list[0])}
            self.score_thr = score_thr
            print('pseudo label loaded!')
        elif pseudo_file is not None :
            print('loading pseudo label from %s'%pseudo_file)
            self.pseudo_ann = load(pseudo_file)
            self.img_index = {ann['img_id']: i for i, ann in enumerate(self.pseudo_ann)}
            self.score_thr = score_thr
            print('pseudo label loaded!')
        else:
            self.pseudo_ann = None
        super().__init__(*args,
                         seg_map_suffix=seg_map_suffix,
                         proposal_file=proposal_file,
                         file_client_args = file_client_args,
                         backend_args = backend_args,
                         **kwargs)
        # self._metainfo = self._load_metainfo(copy.deepcopy(metainfo))
        

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        with get_local_path(self.ann_file, backend_args=self.backend_args) as local_path:
            self.coco = self.COCOAPI(local_path)
        # The order of returned `cat_ids` will not
        # change with the order of the `classes`
    
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.metainfo['classes'])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)
        img_ids = self.coco.get_img_ids()

        stage_index = self.filter_cfg.get('stage_index',None)
        classes_groups = self.filter_cfg.get('classes_groups',None)


        random.seed(self.seed)
        random.shuffle(img_ids)
        if self.filter_cfg.get('equally_subset',False):
            assert classes_groups is not None ,"classes_groups is None"
            assert type(classes_groups) is list ,"classes_groups is not list"
            assert stage_index is not None,"stage_index is None"
            num_of_imgs_per_phases = len(img_ids) // len(classes_groups)
            img_ids = img_ids[stage_index*num_of_imgs_per_phases : (stage_index+1)*num_of_imgs_per_phases]
        
        subset_imgs_scale = self.filter_cfg.get('subset_imgs_scale',None) #(0,0.5) stage1
        if subset_imgs_scale is not None:
            start_index_imgs = int(len(img_ids) * subset_imgs_scale[0])
            end_index_imgs = int(len(img_ids) * subset_imgs_scale[1])
            img_ids = img_ids[start_index_imgs : end_index_imgs]
        
        filtered_cat_list = self.cat_ids
        random.seed(self.seed)
        random.shuffle(filtered_cat_list)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        if self.filter_cfg.get('class_spilt',False):
            assert classes_groups is not None ,"classes_groups is None"
            assert type(classes_groups) is list ,"classes_groups is not list"
            assert stage_index is not None,"stage_index is None"
            filtered_cat_list = filtered_cat_list[sum(classes_groups[:stage_index]):sum(classes_groups[:stage_index+1])]

        data_list = []
        total_ann_ids = []

        for img_id in img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            filtered_ann_info = []
            if self.filter_cfg.get('class_spilt',False):
                for _,ann in enumerate(raw_ann_info):
                    if ann['category_id'] in filtered_cat_list:
                        filtered_ann_info.append(ann)
            else:
                filtered_ann_info=raw_ann_info

            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                filtered_ann_info,
                'raw_img_info':
                raw_img_info
            })
            data_list.append(parsed_data_info)
        
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"
        del self.coco
        test = dict()
        for data in data_list:
            for instance in data['instances']:
                if instance['bbox_label'] in test.keys():
                    test[instance['bbox_label']] +=1
                else:
                    test[instance['bbox_label']] =1
        return data_list

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}

        # TODO: need to change data_prefix['img'] to data_prefix['img_path']
        img_path = osp.join(self.data_prefix['img'], img_info['file_name'])
        if self.data_prefix.get('seg', None):
            seg_map_path = osp.join(
                self.data_prefix['seg'],
                img_info['file_name'].rsplit('.', 1)[0] + self.seg_map_suffix)
        else:
            seg_map_path = None
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[ann['category_id']]

            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']

            instances.append(instance)
        if self.temporal:
            pseudo_index = self.img_index.get(data_info['img_id'],None)
            stage_index = self.filter_cfg.get('stage_index',None)
            class_group = self.filter_cfg.get('classes_groups',None)
            if pseudo_index is not None:
                for temporal_index,pseudo_ann in enumerate(self.pseudo_ann_list):
                    label_thr_down = sum(class_group[:temporal_index])
                    label_thr_up = sum(class_group[:temporal_index+1])
                    pseudo = pseudo_ann[pseudo_index]
                    if type(self.score_thr) is float:
                        score_thr = self.score_thr
                    elif type(self.score_thr) is tuple or type(self.score_thr) is list:
                        mean_score = pseudo['pred_instances']['scores'].mean()
                        std_score = pseudo['pred_instances']['scores'].std()
                        score_thr = self.score_thr[0]*mean_score + self.score_thr[1]*std_score
                    else:
                        raise RuntimeError("error: score_thr is not right")
                    
                    index = torch.where(pseudo['pred_instances']['scores'] > score_thr)[0]

                    if index.shape[0] > 0:
                        bboxes = pseudo['pred_instances']['bboxes'][index].tolist()
                        labels = pseudo['pred_instances']['labels'][index].tolist()
                        scores = pseudo['pred_instances']['scores'][index].tolist()
                        for bbox_p,label_p,score_p in zip(bboxes,labels,scores):
                            if label_p < label_thr_up and label_p >= label_thr_down:
                                instance = {}
                                instance['bbox'] = [round(i, 2) for i in bbox_p]
                                instance['bbox_label'] = label_p
                                instance['ignore_flag'] = 0
                                instance['score'] = round(score_p,4)
                                instances.append(instance)

        elif self.pseudo_ann is not None:
            pseudo_index = self.img_index.get(data_info['img_id'],None)
            stage_index = self.filter_cfg.get('stage_index',None)
            class_group = self.filter_cfg.get('classes_groups',None)
            if stage_index is not None and class_group is not None:
                label_thr = sum(class_group[:stage_index])
            if pseudo_index is not None:
                pseudo = self.pseudo_ann[pseudo_index]

                if type(self.score_thr) is float:
                    score_thr = self.score_thr
                elif type(self.score_thr) is tuple or type(self.score_thr) is list:
                    mean_score = pseudo['pred_instances']['scores'].mean()
                    std_score = pseudo['pred_instances']['scores'].std()
                    score_thr = self.score_thr[0]*mean_score + self.score_thr[1]*std_score
                else:
                    raise RuntimeError("error: score_thr is not right")
                
                index = torch.where(pseudo['pred_instances']['scores'] > score_thr)[0]

                if index.shape[0] > 0:
                    bboxes = pseudo['pred_instances']['bboxes'][index].tolist()
                    labels = pseudo['pred_instances']['labels'][index].tolist()
                    scores = pseudo['pred_instances']['scores'][index].tolist()
                    for bbox_p,label_p,score_p in zip(bboxes,labels,scores):
                        if label_p < label_thr:
                            instance = {}
                            instance['bbox'] = [round(i, 2) for i in bbox_p]
                            instance['bbox_label'] = label_p
                            instance['ignore_flag'] = 0
                            instance['score'] = round(score_p,4)
                            instances.append(instance)

        data_info['instances'] = instances
        return data_info

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        if self.filter_cfg is None:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
        min_size = self.filter_cfg.get('min_size', 0)

        # obtain images that contain annotation
        ids_with_ann = set(data_info['img_id'] for data_info in self.data_list)
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            img_id = data_info['img_id']
            width = data_info['width']
            height = data_info['height']
            if filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(width, height) >= min_size:
                valid_data_infos.append(data_info)

        return valid_data_infos