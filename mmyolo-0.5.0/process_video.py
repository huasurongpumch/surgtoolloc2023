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
execute_in_docker = False


class VideoLoader():
    def load(self, *, fname):
        path = Path(fname)
        print('File found: ' + str(path))
        if ((str(path)[-3:])) == 'mp4':
            if not path.is_file():
                raise IOError(
                    f"Could not load {fname} using {self.__class__.__qualname__}."
                )
                # cap = cv2.VideoCapture(str(fname))
            # return [{"video": cap, "path": fname}]
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
                    # UniqueVideoValidator(),
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
        self.show_dir = show_dir
        self.final_score_thr = final_score_thr

    def process_case(self, *, idx, case):
        # Input video would return the collection of all frames (cap object)
        input_video_file_path = case  # VideoLoader.load(case)
        # Detect and score candidates
        scored_candidates = self.predict(case.path)  # video file > load evalutils.py

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
        for i in range(len(labels)):
            pred_class = int(labels[i].detach())
            score = float(scores[i].detach())
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
