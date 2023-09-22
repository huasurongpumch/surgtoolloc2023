import torch
from mmengine.runner.checkpoint import load_checkpoint,_load_checkpoint
a = _load_checkpoint('/data1/surgtoolloc/pretrain/swin_base_patch4_window7_224_22k.pth', 'cpu')
b = _load_checkpoint('/data1/surgtoolloc/pretrain/simmim_swin-base-w6_2xb256-amp-coslr-800e_mia-192px.pth', 'cpu')
c = _load_checkpoint('/data1/surgtoolloc/mmyolo_0.5.0/work_dirs/rtmdet_l_syncbn_fast_8xb32-80e_size864_swinbase_224/fold_0/best_coco_bbox_mAP_epoch_62.pth','cpu')
print(1)