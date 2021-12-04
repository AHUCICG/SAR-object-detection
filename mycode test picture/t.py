from mmdet.apis import init_detector, inference_detector, show_result_pyplot

config_file = '/home/ahu-xiarunfan/下载/Swin-Transformer-Object-Detection-master/tools/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py'
checkpoint_file = '/home/ahu-xiarunfan/下载/Swin-Transformer-Object-Detection-master/tools/work_dirs/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco/epoch_100.pth'
# 初始化模型
model = init_detector(config_file, checkpoint_file)
# 测试一张图片
img = '/home/ahu-xiarunfan/下载/Swin-Transformer-Object-Detection-master/test1.jpg'
# result = inference_detector(model, img)
show_result_pyplot(model, img, result="bbox")
