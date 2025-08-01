# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Sequence

from mmengine.evaluator import DumpResults
from mmengine.evaluator.metric import _to_cpu

from mmdet.registry import METRICS
from mmdet.structures.mask import encode_mask_results

import logging
from abc import ABCMeta, abstractmethod
from typing import Any, List, Optional, Sequence, Union

from torch import Tensor

from mmengine.dist import (broadcast_object_list, collect_results,
                           is_main_process)
from mmengine.fileio import dump
from mmengine.logging import print_log
from mmengine.structures import BaseDataElement


@METRICS.register_module()
class DumpDetResultsTTA(DumpResults):
    """Dump model predictions to a pickle file for offline evaluation.

    Different from `DumpResults` in MMEngine, it compresses instance
    segmentation masks into RLE format.

    Args:
        out_file_path (str): Path of the dumped file. Must end with '.pkl'
            or '.pickle'.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
    """

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """transfer tensors in predictions to CPU."""
        data_samples = _to_cpu(data_samples)
        # for data_sample in data_samples:
        #     # remove gt
        #     import pdb;pdb.set_trace()
        #     # data_sample.pop('gt_instances', None)
        #     # data_sample.pop('ignored_instances', None)
        #     # data_sample.pop('gt_panoptic_seg', None)

        #     if 'pred_instances' in data_sample:
        #         pred = data_sample['pred_instances']
        #         # encode mask to RLE
        #         if 'masks' in pred:
        #             pred['masks'] = encode_mask_results(pred['masks'].numpy())
        #     if 'pred_panoptic_seg' in data_sample:
        #         warnings.warn(
        #             'Panoptic segmentation map will not be compressed. '
        #             'The dumped file will be extremely large! '
        #             'Suggest using `CocoPanopticMetric` to save the coco '
        #             'format json and segmentation png files directly.')
        # import pdb;pdb.set_trace()
        self.results.extend(data_samples)

    def evaluate(self, size: int) -> dict:
        """Evaluate the model performance of the whole dataset after processing
        all batches.

        Args:
            size (int): Length of the entire validation dataset. When batch
                size > 1, the dataloader may pad some data samples to make
                sure all ranks have the same length of dataset slice. The
                ``collect_results`` function will drop the padded data based on
                this size.

        Returns:
            dict: Evaluation metrics dict on the val dataset. The keys are the
            names of the metrics, and the values are corresponding results.
        """
        size = len(self.results)
        if len(self.results) == 0:
            print_log(
                f'{self.__class__.__name__} got empty `self.results`. Please '
                'ensure that the processed results are properly added into '
                '`self.results` in `process` method.',
                logger='current',
                level=logging.WARNING)

        if self.collect_device == 'cpu':
            results = collect_results(
                self.results,
                size,
                self.collect_device,
                tmpdir=self.collect_dir)
        else:
            results = collect_results(self.results, size, self.collect_device)

        if is_main_process():
            # cast all tensors in results list to cpu
            results = _to_cpu(results)
            _metrics = self.compute_metrics(results)  # type: ignore
            # Add prefix to metric names
            if self.prefix:
                _metrics = {
                    '/'.join((self.prefix, k)): v
                    for k, v in _metrics.items()
                }
            metrics = [_metrics]
        else:
            metrics = [None]  # type: ignore

        broadcast_object_list(metrics)

        # reset the results list
        self.results.clear()
        return metrics[0]