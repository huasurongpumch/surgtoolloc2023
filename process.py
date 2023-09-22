import SimpleITK
import numpy as np
import cv2
from pandas import DataFrame
from pathlib import Path
from scipy.ndimage import center_of_mass, label
from pathlib import Path
from evalutils import DetectionAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    DataFrameValidator,
)
from typing import (Tuple)
from evalutils.exceptions import ValidationError
import random
import json
import os
from argparse import ArgumentParser
from pathlib import Path

import mmcv
from mmdet.apis import inference_detector, init_detector
from mmengine.config import Config, ConfigDict
from mmengine.logging import print_log
from mmengine.utils import ProgressBar, path

from mmyolo.registry import VISUALIZERS
from mmyolo.utils import switch_to_deploy
from mmyolo.utils.labelme_utils import LabelmeFormat
from mmyolo.utils.misc import get_file_list, show_data_classes
import os
import sys
import argparse
sys.path.append('/OSTrack')
from ensemble_boxes import *
from bbox_utils import xywh2xyxy, xyxy2xywh, clip_bbox
from lib.test.evaluation import Tracker
device = 'cuda:0'

model_dir = '/mmyolo/my_docker/'
config = [model_dir + 'rtml/rtm_l.py',]
checkpoint = [model_dir + 'rtml/rtm_l.pth',]
score_thr = [0.05, 0.05, 0.05,0.05,0.05]
wbf_iou_thr = 0.5
#-------------track-----------------------
use_sot = False
tracker_name = 'ostrack'
tracker_param = 'vitb_384_mae_ce_32x4_ep300'

sot_corr_thr = 0.05#不用
sot_score_thr = 0.5
final_score_thr = 0.05
sot_anddet_wbf_weights = [3,1]
####
# Toggle the variable below to debug locally. The final container would need to have execute_in_docker=True
####
execute_in_docker = True


class VideoLoader():
    def load(self, *, fname):
        path = Path(fname)
        print('File found: ' + str(path))
        if ((str(path)[-3:])) == 'mp4':
            if not path.is_file():
                raise IOError(
                    f"Could not load {fname} using {self.__class__.__qualname__}."
                )
                #cap = cv2.VideoCapture(str(fname))
            #return [{"video": cap, "path": fname}]
            return [{"path": fname}]

# only path valid
    def hash_video(self, input_video):
        pass


class UniqueVideoValidator(DataFrameValidator):
    """
    Validates that each video in the set is unique
    """

    def validate(self, *, df: DataFrame):
        try:
            hashes = df["video"]
        except KeyError:
            raise ValidationError("Column `video` not found in DataFrame.")

        if len(set(hashes)) != len(hashes):
            raise ValidationError(
                "The videos are not unique, please submit a unique video for "
                "each case."
            )

class Surgtoolloc_det(DetectionAlgorithm):
    def __init__(self):
        super().__init__(
            index_key='input_video',
            file_loaders={'input_video': VideoLoader()},
            input_path=Path("/input/") if execute_in_docker else Path("./test/"),
            output_file=Path("/output/surgical-tools.json") if execute_in_docker else Path(
                            "./output/surgical-tools.json"),
            validators=dict(
                input_video=(
                    #UniqueVideoValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )
        
        ###                                                                                                     ###
        ###  TODO: adapt the following part for creating your model and loading weights
        ###                                                                                                     ###

        self.tool_list = [
            "bipolar_dissector",
            "bipolar_forceps",
            "cadiere_forceps",
            "clip_applier",
            "force_bipolar",
            "grasping_retractor",
            "monopolar_curved_scissor",
            "needle_driver",
            "permanent_cautery_hook_spatula",
            "prograsp_forceps",
            "stapler",
            "suction_irrigator",
            "tip_up_fenestrated_grasper",
            "vessel_sealer",
        ]
        self.model = {}
        self.score_thr = {}
        for i in range(len(config)):
            self.model[i] = init_detector(config[i], checkpoint[i], device=device, cfg_options={})
            self.score_thr[i] = score_thr[i]


        if use_sot:
            self.tracker_name = tracker_name
            self.tracker_param = tracker_param
            self.surgtracker = Tracker(tracker_name, tracker_param, "video")
            self.sot_corr_thr = sot_corr_thr
            self.sot_score_thr = sot_score_thr
            self.sot_anddet_wbf_weights = sot_anddet_wbf_weights
            self.pre_frame = None
            self.pre_bboxes = None
            self.pre_scores = None
            self.pre_labels = None

        self.final_score_thr = final_score_thr
    def process_case(self, *, idx, case):
        # Input video would return the collection of all frames (cap object)
        input_video_file_path = case #VideoLoader.load(case)
        # Detect and score candidates
        scored_candidates = self.predict(case.path) #video file > load evalutils.py

        # Write resulting candidates to result.json for this case
        return dict(type="Multiple 2D bounding boxes", boxes=scored_candidates, version={"major": 1, "minor": 0})

    def save(self):
        with open(str(self._output_file), "w") as f:
            json.dump(self._case_results[0], f)

    def generate_bbox(self, frame_id, frame):
        # bbox coordinates are the four corners of a box: [x, y, 0.5]
        # Starting with top left as first corner, then following the clockwise sequence
        # origin is defined as the top left corner of the video frame
        predictions = []
        bboxes_list = []
        labels_list = []
        scores_list = []
        h, w, _ = frame.shape

        for key in self.model.keys():
            data_sample = inference_detector(self.model[key], frame)
            data_sample = data_sample.cpu()
            pred_instances = data_sample.pred_instances
            bboxes = pred_instances.bboxes
            labels = pred_instances.labels
            scores = pred_instances.scores
            #-------------------------处理bboxes-------------------------------------
            bboxes = bboxes.numpy()
            labels = labels.numpy()
            scores = scores.numpy()
            inds = scores > self.score_thr[key]
            bboxes = bboxes[inds, :4]
            labels = labels[inds]
            scores = scores[inds]
            #------------
            #-----------------------------------------------------------------------
            bboxes[:, 0::2] /= w
            bboxes[:, 1::2] /= h
            bboxes_list.append(bboxes.tolist())
            labels_list.append(labels.tolist())
            scores_list.append(scores.tolist())
        #bboxes, scores, labels = weighted_boxes_fusion(bboxes_list, scores_list, labels_list, weights=None,
                                                        #iou_thr=wbf_iou_thr, skip_box_thr=0.0,conf_type='avg',fusion_type='avg')
        if use_sot and frame_id != 0:
            bboxes_list = []
            labels_list = []
            scores_list = []
            bboxes_list.append(bboxes.tolist())
            labels_list.append(labels.tolist())
            scores_list.append(scores.tolist())

            sot_bboxes = np.empty((0, 4))
            sot_scores = np.empty((0,))
            sot_labels = np.empty((0,))
            for i in range(len(self.pre_labels)):
                box = self.pre_bboxes[i]
                box = xyxy2xywh(box)
                self.surgtracker.init_state_t(self.pre_frame, box)
                rect_pred = self.surgtracker.track_t(frame)
                sot_bbox = xywh2xyxy(rect_pred)
                sot_bbox = clip_bbox(sot_bbox, (w, h))
                sot_label = self.pre_labels[i]
                score = self.pre_scores[i]
                sot_corr = 1.0
                sot_score = sot_corr * score
                if sot_corr > self.sot_corr_thr and sot_score > self.sot_score_thr:
                    sot_bboxes = np.concatenate((sot_bboxes, sot_bbox.reshape((-1, 4))), axis=0)
                    sot_scores = np.concatenate((sot_scores, [sot_score]), axis=0)
                    sot_labels = np.concatenate((sot_labels, [sot_label]), axis=0)
            sot_bboxes[:, 0::2] /= w
            sot_bboxes[:, 1::2] /= h
            bboxes_list.append(sot_bboxes.tolist())
            labels_list.append(sot_labels.tolist())
            scores_list.append(sot_scores.tolist())
            bboxes, scores, labels = weighted_boxes_fusion(bboxes_list, scores_list, labels_list, weights=self.sot_anddet_wbf_weights,
                                                           iou_thr=wbf_iou_thr, skip_box_thr=0.0,conf_type='max',fusion_type = 'avg')
        bboxes[:, 0::2] *= w
        bboxes[:, 1::2] *= h
        cut_idx = len(scores)
        for i in range(len(scores)):
            if scores[i] < self.final_score_thr:
                cut_idx = i
                break
        bboxes = bboxes[:cut_idx]
        scores = scores[:cut_idx]
        labels = labels[:cut_idx]


        if use_sot:
            self.pre_frame = frame.copy()
            self.pre_bboxes = bboxes.copy()
            self.pre_scores = scores.copy()
            self.pre_labels = labels.copy()

        predictions = []
        for i in range(len(labels)):
            pred_class = int(labels[i])
            score = float(scores[i])
            name = f'slice_nr_{frame_id}_' + self.tool_list[pred_class]
            pred_bbox = bboxes[i].tolist()
            bbox = [[pred_bbox[0], pred_bbox[1], 0.5],
                    [pred_bbox[2], pred_bbox[1], 0.5],
                    [pred_bbox[2], pred_bbox[3], 0.5],
                    [pred_bbox[0], pred_bbox[3], 0.5]]
            prediction = {"corners": bbox, "name": name, "probability": score}
            predictions.append(prediction)

        # if self.show_dir is not None:
        #     show_result = []
        #     bboxes = np.array(bboxes)
        #     scores = np.array(scores).reshape((-1, 1))
        #     labels = np.array(labels)
        #     for tool in range(len(self.tool_list)):
        #         result = np.empty((0, 5))
        #         idx = labels == tool
        #         result = np.vstack((result, np.hstack([bboxes[idx], scores[idx]])))
        #         show_result.append(result)
        #
        #     out_file = self.show_dir + f'slice_nr_{frame_id}.jpg'
            # show_result_pyplot(
            #     self.model[0],
            #     frame,
            #     show_result,
            #     palette='coco',
            #     score_thr=self.final_score_thr,
            #     out_file=out_file)
        return predictions

    def predict(self, fname) -> DataFrame:
        """
        Inputs:
        fname -> video file path
        
        Output:
        tools -> list of prediction dictionaries (per frame) in the correct format as described in documentation 
        """
        print('Video file to be loaded: ' + str(fname))
        cap = cv2.VideoCapture(str(fname))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ###                                                                     ###
        ###  TODO: adapt the following part for YOUR submission: make prediction
        ###                                                                     ###

        print('start infering...')
        all_frames_predicted_outputs = []
        # for fid in range(num_frames):
        #     tool_detections = self.generate_bbox(fid)
        #     all_frames_predicted_outputs += tool_detections
        for fid in range(num_frames):
            ret, frame = cap.read()
            if ret:
                tool_detections = self.generate_bbox(fid, frame)
                all_frames_predicted_outputs += tool_detections
        cap.release()
        print('task finished.')
        return all_frames_predicted_outputs


if __name__ == "__main__":
    Surgtoolloc_det().process()
