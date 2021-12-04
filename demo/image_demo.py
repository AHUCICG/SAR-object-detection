from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot


def main():
    parser = ArgumentParser()
    parser.add_argument('--img', default='/home/ahu-xiarunfan/下载/Swin-Transformer-Object-Detection-master/school/001009.jpg', help='test6.jpg')
    parser.add_argument('--config',default='/home/ahu-xiarunfan/下载/Swin-Transformer-Object-Detection-master/tools/faster_rcnn_r50_fpn_1x_coco.py')
    parser.add_argument('--checkpoint', default='/home/ahu-xiarunfan/下载/work_dirs/school/epoch_534.pth')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_detector(model, args.img)
    # show the results
    show_result_pyplot(model, args.img, result, score_thr=args.score_thr)
    # show_result_pyplot(model, args.img, result)




if __name__ == '__main__':
    main()
# Copyright (c) OpenMMLab. All rights reserved.
# import asyncio
# from argparse import ArgumentParser
#
# from mmdet.apis import (async_inference_detector, inference_detector,
#                         init_detector, show_result_pyplot)
#
#
# def parse_args():
#     parser = ArgumentParser()
#     parser.add_argument('img', default='/home/ahu-xiarunfan/下载/Swin-Transformer-Object-Detection-master/test6.jpg')
#     parser.add_argument('config',default='/home/ahu-xiarunfan/下载/Swin-Transformer-Object-Detection-master/tools/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py')
#     parser.add_argument('checkpoint',default='/home/ahu-xiarunfan/下载/work_dirs/4/epoch_145.pth')
#     parser.add_argument(
#         '--device', default='cuda:0', help='Device used for inference')
#     parser.add_argument(
#         '--score-thr', type=float, default=0.3, help='bbox score threshold')
#     parser.add_argument(
#         '--async-test',
#         action='store_true',
#         help='whether to set async options for async inference.')
#     args = parser.parse_args()
#     return args
#
#
# def main(args):
#     # build the model from a config file and a checkpoint file
#     model = init_detector(args.config, args.checkpoint, device=args.device)
#     # test a single image
#     result = inference_detector(model, args.img)
#     # show the results
#     show_result_pyplot(model, args.img, result, score_thr=args.score_thr)
#
#
# async def async_main(args):
#     # build the model from a config file and a checkpoint file
#     model = init_detector(args.config, args.checkpoint, device=args.device)
#     # test a single image
#     tasks = asyncio.create_task(async_inference_detector(model, args.img))
#     result = await asyncio.gather(tasks)
#     # show the results
#     show_result_pyplot(model, args.img, result[0], score_thr=args.score_thr)
#
#
# if __name__ == '__main__':
#     args = parse_args()
#     if args.async_test:
#         asyncio.run(async_main(args))
#     else:
#         main(args)
