from pathlib import Path
from yacs.config import CfgNode as CN
import os
import time
import logging

_C = CN()
_C.name = ''
_C.print_freq = 40
_C.workers = 16
_C.log_dir = 'logs'
_C.model_dir = 'ckps'
_C.output_dir='output'
_C.iter= 'iter0'
_C.GPUS=[0]

_C.dataset = 'cifar10'
_C.data_path = './data/cifar10'
_C.num_classes = 100
_C.imb_factor = 0.01
_C.imb_type ='exp'
_C.weighted_sampler='default'
_C.quantity_ratio=1.0
_C.backbone = 'resnet32_fe'
_C.resume = ''
_C.head_class_idx = [0, 1]
_C.med_class_idx = [0, 1]
_C.tail_class_idx = [0, 1]

_C.mode = None
_C.smooth_tail = None
_C.smooth_head = None
_C.shift_bn = False
_C.lr_factor = None
_C.lr = 0.1
_C.batch_size = 128
_C.weight_decay = 0.002
_C.num_epochs = 200
_C.momentum = 0.9
_C.cos = False
_C.mixup = 'mixup'
_C.alpha = 1.0

_C.scheduler='default'
_C.drw =160
_C.ppw_min=30
_C.ppw_max=160
_C.loss='CrossEntropy'
_C.s=30
_C.max_m=0.5
_C.ppw_alpha=5
_C.ppmix_epoch_min=30
_C.ppmix_epoch_max=160
_C.gamma=1.0
_C.ppm_alpha=1


def update_config(cfg, args):
    cfg.defrost()
    
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    # cfg.freeze()

def create_logger(cfg, cfg_name):
    time_str = time.strftime('%Y%m%d%H%M')
    #cfg_name = os.path.basename(cfg_name).split('.')[0]

    log_dir = Path(cfg.output_dir)/ Path(cfg.iter)/(cfg.name + '_' + time_str) / Path(cfg.log_dir)
    print('=> creating {}'.format(log_dir))
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = '{}.txt'.format(cfg.name)
    final_log_file = log_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    model_dir = Path(cfg.output_dir)/Path(cfg.iter)/ (cfg.name + '_' + time_str) / Path(cfg.model_dir)
    print('=> creating {}'.format(model_dir))
    model_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(model_dir)