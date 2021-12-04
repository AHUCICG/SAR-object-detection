import os
import mmcv
import numpy as np
import matplotlib.pyplot as plt

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from mmcv import Config
from mmdet.datasets import build_dataset

# MODEL = "mask_rcnn"
# MODEL_NAME = "mask_rcnn_r50_fpn_1x_coco_senet"
# CONFIG_FILE = f"/home/ahu-xiarunfan/下载/Swin-Transformer-Object-Detection-master/tools/cascade_mask_rcnn_swin_small_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py"
# CONFIG_FILE = f"/home/ahu-xiarunfan/下载/Swin-Transformer-Object-Detection-master/tools/faster_rcnn_r50_fpn_1x_coco.py"
# CONFIG_FILE = f"/home/ahu-xiarunfan/下载/Swin-Transformer-Object-Detection-master/tools/yolov3_d53_320_273e_coco.py"
CONFIG_FILE = f"/home/ahu-xiarunfan/下载/Swin-Transformer-Object-Detection-master/configs/nas_fcos/nas_fcos_fcoshead_r50_caffe_fpn_gn-head_4x4_1x_coco.py"


# CONFIG_FILE1 = f"/home/ahu-xiarunfan/下载/Swin-Transformer-Object-Detection-master/tools/faster_rcnn_r50_fpn_1x_coco.py"
CONFIG_FILE1 = f"/home/ahu-xiarunfan/下载/Swin-Transformer-Object-Detection-master/tools/cascade_mask_rcnn_swin_small_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py"

RESULT_FILE1 = f"/home/ahu-xiarunfan/下载/Swin-Transformer-Object-Detection-master/poker_results.pkl"
# RESULT_FILE = f"/home/ahu-xiarunfan/下载/Swin-Transformer-Object-Detection-master/faster1.pkl"
RESULT_FILE = f"/home/ahu-xiarunfan/下载/Swin-Transformer-Object-Detection-master/swin.pkl"
# RESULT_FILE = f"/home/ahu-xiarunfan/下载/Swin-Transformer-Object-Detection-master/fasterrcnn.pkl"
# RESULT_FILE = f"/home/ahu-xiarunfan/下载/Swin-Transformer-Object-Detection-master/yolov3.pkl"
RESULT_FILE = f"/home/ahu-xiarunfan/下载/Swin-Transformer-Object-Detection-master/nasfoc.pkl"





def plot_pr_curve(config_file, config_file1,result_file, result_file1,metric="bbox"):
    """plot precison-recall curve based on testing results of pkl file.

        Args:
            config_file (list[list | tuple]): config file path.
            result_file (str): pkl file of testing results path.
            metric (str): Metrics to be evaluated. Options are
                'bbox', 'segm'.
    """

    cfg = Config.fromfile(config_file)
    cfg1 = Config.fromfile(config_file1)
    # turn on test mode of dataset
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True

    # build dataset
    dataset = build_dataset(cfg.data.test)
    # dataset1 = build_dataset(cfg1.data.test)

    # load result file in pkl format
    pkl_results = mmcv.load(result_file)
    # pkl_results1 = mmcv.load(result_file1)

    # convert pkl file (list[list | tuple | ndarray]) to json
    json_results, _ = dataset.format_results(pkl_results)
    # json_results1, _ = dataset1.format_results(pkl_results1)
    # initialize COCO instance
    coco = COCO(annotation_file=cfg.data.test.ann_file)
    # coco1 = COCO(annotation_file=cfg1.data.test.ann_file)
    coco_gt = coco
    # coco_gt1 = coco1
    coco_dt = coco_gt.loadRes(json_results[metric])
    # coco_dt1 = coco_gt.loadRes(json_results1[metric])
    # coco_dt = COCO()
    # initialize COCOeval instance
    coco_eval = COCOeval(coco_gt, coco_dt, metric)
    # coco_eval1 = COCOeval(coco_gt, coco_dt1, metric)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    # coco_eval1.evaluate()
    # coco_eval1.accumulate()
    # coco_eval1.summarize()
    # extract eval data
    precisions = coco_eval.eval["precision"]
    # precisions1 = coco_eval1.eval["precision"]
    '''
    precisions[T, R, K, A, M]
    T: iou thresholds [0.5 : 0.05 : 0.95], idx from 0 to 9
    R: recall thresholds [0 : 0.01 : 1], idx from 0 to 100
    K: category, idx from 0 to ...
    A: area range, (all, small, medium, large), idx from 0 to 3 
    M: max dets, (1, 10, 100), idx from 0 to 2
    '''
    # pr_array1 = precisions[0, :, 0, 0, 2]
    # pr_array2 = precisions[1, :, 0, 0, 2]
    # pr_array3 = precisions[2, :, 0, 0, 2]
    # pr_array4 = precisions[3, :, 0, 0, 2]
    # pr_array5 = precisions[4, :, 0, 0, 2]
    # pr_array6 = precisions[5, :, 0, 0, 2]
    # pr_array7 = precisions[6, :, 0, 0, 2]
    # pr_array8 = precisions[7, :, 0, 0, 2]
    # pr_array9 = precisions[8, :, 0, 0, 2]
    # pr_array10 = precisions[9, :, 0, 0, 2]


    pr_array1 = precisions[0, :, 0, 0, 2]
    print(pr_array1)
    # pr_array2 = precisions[0, :, 0, 0, 2]
    # print(pr_array2)
    # pr_array3 = precisions[2, :, 0, 0, 2]
    # pr_array4 = precisions[3, :, 0, 0, 2]
    # pr_array5 = precisions[4, :, 0, 0, 2]
    # pr_array6 = precisions[5, :, 0, 0, 2]
    # pr_array7 = precisions[6, :, 0, 0, 2]
    # pr_array8 = precisions[7, :, 0, 0, 2]
    # pr_array9 = precisions[8, :, 0, 0, 2]
    # pr_array10 = precisions[9, :, 0, 0, 2]

    x = np.arange(0.0, 1.01, 0.01)
    # plot PR curve
    plt.plot(x, pr_array1, label="iou=0.5")
    # plt.plot(x, pr_array2, label="iou=0.55")
    # plt.plot(x, pr_array3, label="iou=0.6")
    # plt.plot(x, pr_array4, label="iou=0.65")
    # plt.plot(x, pr_array5, label="iou=0.7")
    # plt.plot(x, pr_array6, label="iou=0.75")
    # plt.plot(x, pr_array7, label="iou=0.8")
    # plt.plot(x, pr_array8, label="iou=0.85")
    # plt.plot(x, pr_array9, label="iou=0.9")
    # plt.plot(x, pr_array10, label="iou=0.95")

    plt.xlabel("recall")
    plt.ylabel("precison")
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.01)
    plt.grid(True)
    plt.legend(loc="lower left")
    plt.show()


if __name__ == "__main__":
    # plot_pr_curve(config_file=CONFIG_FILE,config_file1=CONFIG_FILE1, result_file=RESULT_FILE,result_file1=RESULT_FILE1, metric="bbox")
    plot_pr_curve(config_file=CONFIG_FILE,config_file1=CONFIG_FILE1, result_file=RESULT_FILE,result_file1=RESULT_FILE1, metric="bbox")
