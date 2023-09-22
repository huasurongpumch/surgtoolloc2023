# import SimpleITK
import numpy as np
import cv2
# from pandas import DataFrame
# from pathlib import Path
# from scipy.ndimage import center_of_mass, label
# from pathlib import Path
# from evalutils import DetectionAlgorithm
# from evalutils.validators import (
#     UniquePathIndicesValidator,
#     DataFrameValidator,
# )
# from typing import (Tuple)
# from evalutils.exceptions import ValidationError
# import random
# import json
# import os
# from argparse import ArgumentParser
# from pathlib import Path

import mmcv
from mmdet.apis import inference_detector, init_detector
from mmengine.config import Config, ConfigDict
from mmengine.logging import print_log
from mmengine.utils import ProgressBar, path

from mmyolo.registry import VISUALIZERS
from mmyolo.utils import switch_to_deploy
from mmyolo.utils.labelme_utils import LabelmeFormat
from mmyolo.utils.misc import get_file_list, show_data_classes
import json
def dete(frame_id, frame):
    predictions = []
    bboxes_list = []
    labels_list = []
    scores_list = []
    for key in model.keys():
        data_sample = inference_detector(model[key], frame)
        data_sample = data_sample.cpu()
        pred_instances = data_sample.pred_instances
        bboxes = pred_instances.bboxes
        labels = pred_instances.labels
        scores = pred_instances.scores
    for i in range(len(labels)):
        pred_class = int(labels[i].detach())
        score = float(scores[i].detach())
        name = f'slice_nr_{frame_id}_' + tool_list[pred_class]
        pred_bbox = bboxes[i].tolist()
        bbox = [[pred_bbox[0], pred_bbox[1], 0.5],
                [pred_bbox[2], pred_bbox[1], 0.5],
                [pred_bbox[2], pred_bbox[3], 0.5],
                [pred_bbox[0], pred_bbox[3], 0.5]]
        prediction = {"corners": bbox, "name": name, "probability": score}
        predictions.append(prediction)
    return predictions
device = 'cuda:0'

model_dir = '/data1/surgtoolloc/mmyolo_0.5.0'
config = [model_dir + '/work_dirs/rtmdet_l_syncbn_fast_8xb32-300e_size864/fold_0/rtmdet_l_syncbn_fast_8xb32-300e_size864.py']
checkpoint = [model_dir + '/work_dirs/rtmdet_l_syncbn_fast_8xb32-300e_size864/fold_0/best_coco_bbox_mAP_epoch_70.pth']
score_thr = [0.05, 0.05, 0.05]
show_dir = None  # '/data1/surgtoolloc2022-category-2-main/output/show/'
final_score_thr = 0.05
####
# Toggle the variable below to debug locally. The final container would need to have execute_in_docker=True
####
                                                                                     ###

tool_list = [
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
model = {}
score_thr = {}
model[0] = init_detector(config[0], checkpoint[0], device=device, cfg_options={})
score_thr[0] = 0.5
fname ='/data1/surgtoolloc/mmyolo_0.5.0/test_video/vid_1_short.mp4'
all_frames_predicted_outputs = []
cap = cv2.VideoCapture(str(fname))
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
for fid in range(num_frames):
    ret, frame = cap.read()
    if ret:
        tool_detections = dete(fid, frame)
        all_frames_predicted_outputs += tool_detections

a = dict(type="Multiple 2D bounding boxes", boxes=all_frames_predicted_outputs, version={"major": 1, "minor": 0})
with open('test-1.json', "w") as f:
    json.dump(a, f)

