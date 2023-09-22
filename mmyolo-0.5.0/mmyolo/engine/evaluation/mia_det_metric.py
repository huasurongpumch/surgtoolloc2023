import datetime
import os
import tempfile
from collections import OrderedDict
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.fileio import FileClient
from mmengine.fileio import dump
from mmengine.fileio import load
from mmengine.logging import MMLogger
from mmdet.datasets.api_wrappers import COCO
from mmdet.evaluation.functional import bbox_overlaps

from mmyolo.registry import METRICS


@METRICS.register_module()
class MIADetMetric(BaseMetric):
    def __init__(
            self,
            ann_file: Optional[str] = None,
            classwise: bool = True,
            iou_thr: float = 0.5,
            outfile_prefix: Optional[str] = None,
            file_client_args: dict = dict(backend="disk"),
            collect_device: str = "cpu",
            prefix: Optional[str] = None,
            sort_categories: bool = False,
            score_thr: float = 0.0,
    ):
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.classwise = classwise
        self.score_thr = score_thr
        self.iou_thr = iou_thr
        self.outfile_prefix = outfile_prefix

        self.file_client_args = file_client_args
        self.file_client = FileClient(**file_client_args)

        if ann_file is not None:
            with self.file_client.get_local_path(ann_file) as local_path:
                self._coco_api = COCO(local_path)
                if sort_categories:
                    # `categories` list in objects365_train.json and
                    # objects365_val.json is inconsistent, need sort
                    # list(or dict) before get cat_ids.
                    cats = self._coco_api.cats
                    sorted_cats = {i: cats[i] for i in sorted(cats)}
                    self._coco_api.cats = sorted_cats
                    categories = self._coco_api.dataset["categories"]
                    sorted_categories = sorted(categories, key=lambda i: i["id"])
                    self._coco_api.dataset["categories"] = sorted_categories
        else:
            self._coco_api = None

        self.cat_ids = None
        self.img_ids = None

    def process(self, data_batch: dict, data_samples: Sequence[dict]):
        for data_sample in data_samples:
            result = dict()
            pred = data_sample["pred_instances"]
            result["img_id"] = data_sample["img_id"]
            result["bboxes"] = pred["bboxes"].cpu().numpy()
            result["scores"] = pred["scores"].cpu().numpy()
            result["labels"] = pred["labels"].cpu().numpy()

            gt = dict()
            gt["width"] = data_sample["ori_shape"][1]
            gt["height"] = data_sample["ori_shape"][0]
            gt["img_id"] = data_sample["img_id"]
            if self._coco_api is None:
                # TODO: Need to refactor to support LoadAnnotations
                assert "instances" in data_sample, "ground truth is required for evaluation when `ann_file` is not provided"

                gt["anns"] = data_sample["instances"]
            self.results.append((gt, result))

    def gt_to_coco_json(
            self,
            gt_dicts: Sequence[dict],
            outfile_prefix: str,
    ) -> str:
        categories = [
            dict(id=id, name=name)
            for id, name in enumerate(self.dataset_meta["classes"])
        ]
        image_infos = []
        annotations = []

        for idx, gt_dict in enumerate(gt_dicts):
            img_id = gt_dict.get("img_id", idx)
            image_info = dict(
                id=img_id,
                width=gt_dict["width"],
                height=gt_dict["height"],
                file_name="",
            )
            image_infos.append(image_info)
            for ann in gt_dict["anns"]:
                label = ann["bbox_label"]
                bbox = ann["bbox"]
                coco_bbox = [
                    bbox[0],
                    bbox[1],
                    bbox[2] - bbox[0],
                    bbox[3] - bbox[1],
                ]

                annotation = dict(
                    id=len(annotations) + 1,  # coco api requires id starts with 1
                    image_id=img_id,
                    bbox=coco_bbox,
                    iscrowd=ann.get("ignore_flag", 0),
                    category_id=int(label),
                    area=coco_bbox[2] * coco_bbox[3],
                )
                annotations.append(annotation)

        info = dict(
            date_created=str(datetime.datetime.now()),
            description="Coco json file converted by MIADetMetric.",
        )
        coco_json = dict(
            info=info,
            images=image_infos,
            categories=categories,
            licenses=None,
        )
        if len(annotations) > 0:
            coco_json["annotations"] = annotations
        converted_json_path = f"{outfile_prefix}.gt.json"
        dump(coco_json, converted_json_path)
        return converted_json_path

    def xyxy2xywh(self, bbox: np.ndarray) -> list:
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox: List = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def results2json(
            self,
            results: Sequence[dict],
            outfile_prefix: str,
    ) -> dict:
        bbox_json_results = []
        for idx, result in enumerate(results):
            image_id = result.get("img_id", idx)
            labels = result["labels"]
            bboxes = result["bboxes"]
            scores = result["scores"]
            # bbox results
            for i, label in enumerate(labels):
                data = dict()
                data["image_id"] = image_id
                data["bbox"] = self.xyxy2xywh(bboxes[i])
                data["score"] = float(scores[i])
                data["category_id"] = self.cat_ids[label]
                bbox_json_results.append(data)

        result_files = dict()
        result_files["bbox"] = f"{outfile_prefix}.bbox.json"
        dump(bbox_json_results, result_files["bbox"])

        return result_files

    def compute_metrics(self, results: list) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()

        gts, preds = zip(*results)

        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = os.path.join(tmp_dir.name, "results")
        else:
            outfile_prefix = self.outfile_prefix

        if self._coco_api is None:
            # use converted gt json file to initialize coco api
            logger.info("Converting ground truth to coco format...")
            coco_json_path = self.gt_to_coco_json(
                gt_dicts=gts,
                outfile_prefix=outfile_prefix,
            )
            self._coco_api = COCO(coco_json_path)

        # handle lazy init
        if self.cat_ids is None:
            self.cat_ids = self._coco_api.get_cat_ids(cat_names=self.dataset_meta["classes"])
        if self.img_ids is None:
            self.img_ids = self._coco_api.get_img_ids()

        # convert predictions to coco format and dump to json file
        result_files = self.results2json(preds, outfile_prefix)

        eval_results = OrderedDict()
        logger.info(f"results are saved in {os.path.dirname(outfile_prefix)}")

        logger.info(f"Evaluating bbox...")

        if "bbox" not in result_files:
            raise KeyError(f"`bbox` is not in results")
        try:
            predictions = load(result_files["bbox"])
            coco_dt = self._coco_api.loadRes(predictions)
        except IndexError:
            logger.error("The testing results of the whole dataset is empty.")

            for idx, class_name in enumerate(self.dataset_meta["classes"]):
                eval_results[class_name + "/recall"] = 0
                eval_results[class_name + "/precision"] = 0
                eval_results[class_name + "/num"] = 0

            eval_results["avg/recall"] = 0
            eval_results["avg/precision"] = 0
            eval_results["avg/num"] = 0
            eval_results["f1"] = 0
            return eval_results

        total_num = [0 for _ in range(len(self.cat_ids))]
        recall_num = [0 for _ in range(len(self.cat_ids))]
        pred_num = [0 for _ in range(len(self.cat_ids))]

        for img_id in self._coco_api.get_img_ids():
            gt_boxes = []
            for item in self._coco_api.load_anns(self._coco_api.get_ann_ids(img_id)):
                x, y, w, h = item["bbox"]
                # if item["iscrowd"] == 1:
                #     continue
                gt_boxes.append([x, y, x + w, y + h, item["category_id"], 1])

            gt_boxes = np.array(gt_boxes, dtype=np.float32).reshape(-1, 6)

            for idx, item in enumerate(sorted(self.cat_ids)):
                total_num[idx] += len(gt_boxes[gt_boxes[:, -2] == item])

            pred_boxes = []
            for item in coco_dt.loadAnns(coco_dt.getAnnIds(img_id)):
                x, y, w, h = item["bbox"]
                pred_boxes.append([x, y, x + w, y + h, item["category_id"], item["score"]])

            if len(pred_boxes) == 0:
                continue

            pred_boxes = np.array(pred_boxes, dtype=np.float32).reshape(-1, 6)
            pred_boxes = pred_boxes[pred_boxes[:, -1] > self.score_thr]
            pred_boxes = np.array(sorted(pred_boxes, key=lambda x: x[-1], reverse=True)).reshape(-1, 6)

            gt_match_idx = [-1 for _ in range(len(gt_boxes))]
            pred_match_idx = [-1 for _ in range(len(pred_boxes))]
            for pred_idx, pred_box in enumerate(pred_boxes):
                if pred_box[-1] < self.score_thr:
                    continue

                iou = bbox_overlaps(
                    pred_box[:4].reshape(-1, 4),
                    gt_boxes[:, :4].reshape(-1, 4),
                )[0]
                try:
                    if iou.max() < self.iou_thr:
                        continue
                except:
                    print(len(gt_boxes))
                sorted_inds = iou.argsort()[::-1]
                sorted_gt_boxes = gt_boxes[sorted_inds, :]
                sorted_iou = iou[sorted_inds]
                for gt_idx, gt_box in enumerate(sorted_gt_boxes):
                    if (
                            gt_box[-2] == pred_box[-2]
                            and sorted_iou[gt_idx] > self.iou_thr
                            and gt_match_idx[sorted_inds[gt_idx]] == -1
                    ):
                        gt_match_idx[sorted_inds[gt_idx]] = pred_idx
                        pred_match_idx[pred_idx] = sorted_inds[gt_idx]
                        break

            for gt_idx, gt_box in enumerate(gt_boxes):
                if int(gt_box[-2]) == 0:
                    pass
                    # print(1)
                # total_num[int(gt_box[-2]) - 1] += 1
                if gt_match_idx[gt_idx] != -1:
                    recall_num[int(gt_box[-2])] += 1

            for pred_box, pred_box in enumerate(pred_boxes):
                pred_num[int(pred_box[-2])] += 1

        for idx, class_name in enumerate(self.dataset_meta["classes"]):
            eval_results[class_name + "/recall"] = round(recall_num[idx] / (total_num[idx] + 1e-5), 4)
            eval_results[class_name + "/precision"] = round(recall_num[idx] / (pred_num[idx] + 1e-5), 4)
            eval_results[class_name + "/num"] = total_num[idx]

        eval_results["avg/recall"] = round(sum(recall_num) / (sum(total_num) + 1e-5), 4)
        eval_results["avg/precision"] = round(sum(recall_num) / (sum(pred_num) + 1e-5), 4)
        eval_results["avg/num"] = sum(total_num)
        eval_results["f1"] = 2 * eval_results["avg/recall"] * eval_results["avg/precision"] / (
                    eval_results["avg/recall"] + eval_results["avg/precision"] + 1e-5)
        return eval_results
