# coding=utf-8

from mmdet.apis import init_detector
from mmdet.apis import inference_detector
#from mmdet.apis import show_result_pyplot

# 模型配置文件
config_file = '/home/ahu-xiarunfan/下载/Swin-Transformer-Object-Detection-master/tools/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py'

# 预训练模型文件
checkpoint_file = '/home/ahu-xiarunfan/下载/Swin-Transformer-Object-Detection-master/tools/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco/epoch_100.pth'

# 通过模型配置文件与预训练文件构建模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# 测试单张图片并进行展示
img = 'test1.jpg'
result = inference_detector(model, img)

show_result_pyplot(img, result, model.CLASSES,show=False,out_file=XX)