import numpy as np
import torch
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample
from mmdet.visualization import DetLocalVisualizer
import pickle
import cv2
from mmcv import imread
import mmcv


with open('work_dirs/kmeans_ori.pkl', 'rb') as f:
    pkl = pickle.load(f)

det_local_visualizer = DetLocalVisualizer() 

for data in pkl:
    # 读取 uint 图像
    image_path = data['img_path']
    # image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
    image = imread(image_path)
    image = mmcv.bgr2rgb(image)
    # import pdb;pdb.set_trace()
    gt_det_data_sample = DetDataSample()
    gt_instances = InstanceData()
    gt_instances.bboxes = data['pred_instances']['bboxes']
    gt_instances.labels = data['pred_instances']['labels']
    gt_instances.scores = data['pred_instances']['scores']
    gt_det_data_sample.pred_instances = gt_instances
    gt_det_data_sample.gt_instances = gt_instances
    det_local_visualizer.add_datasample('image', image, gt_det_data_sample,out_file='./work_dirs/out_file.jpg',pred_score_thr=0.3)
    import pdb;pdb.set_trace()
