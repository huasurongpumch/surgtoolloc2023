# Srugtoolloc2023

## Adapting the container to your algorithm
1. First, clone this repository:

```
git clone https://github.com/huasurongpumch/surgtoolloc2023/tree/master
```
2. Download [Our RTMDetL weights]() and put it under `$PROJECT_ROOT$/mmyolo-0.5.0/my_docker/rtml/` named `rtm_l.pth`

3. Our `Dockerfile` should have everything you need

4. Run `build.sh`  to build the container.
 
5. Run `export.sh`. This script will will produce `surgtoolloc_det.tar.gz` (you can change the name of your container by modifying the script). This is the file to be used when uploading the algorithm to Grand Challenge.

## Citation
```
@misc{mmyolo2022,
    title={{MMYOLO: OpenMMLab YOLO} series toolbox and benchmark},
    author={MMYOLO Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmyolo}},
    year={2022}
}
```
```
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}
```
```BibTeX
@misc{2023mmpretrain,
    title={OpenMMLab's Pre-training Toolbox and Benchmark},
    author={MMPreTrain Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmpretrain}},
    year={2023}
}
```
```Bibtex
@inproceedings{ye2022ostrack,
  title={Joint Feature Learning and Relation Modeling for Tracking: A One-Stream Framework},
  author={Ye, Botao and Chang, Hong and Ma, Bingpeng and Shan, Shiguang and Chen, Xilin},
  booktitle={ECCV},
  year={2022}
}
```
