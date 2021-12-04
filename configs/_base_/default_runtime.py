checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# interval = 10

# evaluation = dict(interval=interval, metric='mAP')

custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '../work_dirs/school'

load_from = None
resume_from = None
workflow = [('train', 1)]
