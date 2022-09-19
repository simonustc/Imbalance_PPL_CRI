import argparse
import os
import random
import shutil
import time
import warnings
import numpy as np
import pprint
import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
# from datasets.cifar10 import CIFAR10_LT
from datasets.imagenet import ImageNet_LT
from datasets.ina2018 import iNa2018
from datasets.cao_cifar import CIFAR10_LT
from datasets.cao_cifar100 import CIFAR100_LT
from models import resnet
from models import resnet_places
from models import resnet_cifar
from utils import config, update_config, create_logger
from utils import AverageMeter, ProgressMeter,get_scheduler
from utils import accuracy, calibration
from methods import mixup_data, mixup_criterion
from loss.loss import CrossEntropy, LDAMLoss, FocalLoss, LOWLoss, GHMCLoss, CCELoss,CRILoss
import logging
import torch.backends.cudnn as cudnn
def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='./config/cifar10/cifar10_CE.yaml',
                        required=False,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    update_config(config, args)
    return args


best_acc1 = 0
its_ece = 100

def main():
    args = parse_args()
    best = []
    strGPUs = [str(x) for x in config.GPUS]
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(strGPUs)
    device = torch.device("cuda")

    logger, model_dir = create_logger(config, args.cfg)
    logger.info('\n' + pprint.pformat(args))
    logger.info('\n' + str(config))

    seed = 0
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    main_worker(device,config,logger,model_dir,best)
    logger.handlers.clear()
    logging.shutdown()


def main_worker(device, config, logger, model_dir,best):
        global best_acc1, its_ece

        if config.dataset == 'cifar10' or config.dataset == 'cifar100':
            model = getattr(resnet_cifar, config.backbone)()
            classifier = getattr(resnet_cifar, 'Classifier')(feat_in=64, num_classes=config.num_classes)

        elif config.dataset == 'cifar10_CRI' or config.dataset == 'cifar100_CRI':
            model = getattr(resnet_cifar, config.backbone)()
            classifier = getattr(resnet_cifar, 'Classifier2')(feat_in=64, num_classes=config.num_classes)

        elif config.dataset == 'imagenet':
            model = getattr(resnet, config.backbone)()
            classifier = getattr(resnet, 'Classifier')(feat_in=512, num_classes=config.num_classes)

        elif config.dataset == 'ina2018':
            model = getattr(resnet, config.backbone)()
            classifier = getattr(resnet, 'Classifier')(feat_in=2048, num_classes=config.num_classes)

        elif config.dataset == 'imagenet_CRI':
            model = getattr(resnet, config.backbone)()
            classifier = getattr(resnet, 'Classifier2')(feat_in=512, num_classes=config.num_classes)

        elif config.dataset == 'ina2018_CRI':
            model = getattr(resnet, config.backbone)()
            classifier = getattr(resnet, 'Classifier2')(feat_in=2048, num_classes=config.num_classes)

        model = torch.nn.DataParallel(model).to(device)
        classifier = torch.nn.DataParallel(classifier).to(device)

        if config.resume:
            if os.path.isfile(config.resume):
                logger.info("=> loading checkpoint '{}'".format(config.resume))
                checkpoint = torch.load(config.resume)
                #best_acc1 = best_acc1.to(device)
                model.load_state_dict(checkpoint['state_dict_model'])
                classifier.load_state_dict(checkpoint['state_dict_classifier'])
                logger.info("=> loaded checkpoint '{}' (epoch {})"
                            .format(config.resume, checkpoint['epoch']))
            else:
                logger.info("=> no checkpoint found at '{}'".format(config.resume))

        if config.dataset == 'cifar10' or config.dataset == 'cifar10_CRI':
            dataset = CIFAR10_LT(root=config.data_path, quantity_ratio=config.quantity_ratio,imb_type=config.imb_type,imb_factor=config.imb_factor,weighted_sampler=config.weighted_sampler,
                                 batch_size=config.batch_size, num_works=config.workers)

        elif config.dataset == 'cifar100' or config.dataset == 'cifar100_CRI':
            dataset = CIFAR100_LT(root=config.data_path, quantity_ratio=config.quantity_ratio,imb_type=config.imb_type,imb_factor=config.imb_factor,
                                  batch_size=config.batch_size, num_works=config.workers)

        elif config.dataset == 'imagenet' or config.dataset =='imagenet_CRI':
            dataset = ImageNet_LT(root=config.data_path,
                                  batch_size=config.batch_size, num_works=config.workers)

        elif config.dataset == 'ina2018' or config.dataset =='ina2018_CRI':
            dataset = iNa2018(root=config.data_path,
                              batch_size=config.batch_size, num_works=config.workers)

        train_loader = dataset.train_instance
        val_loader = dataset.eval
        num_class_list=dataset.cls_num_list


        para_dict = {
            "num_class_list": num_class_list,
            "config": config,
            "device":device,
        }

        criterion = eval(config.loss)(para_dict=para_dict)

        acc1, ece,head_acc,med_acc,tail_acc = validate(val_loader,device, model, classifier, criterion, config, logger)

        logging.shutdown()


def validate(val_loader,device,model, classifier, criterion, config, logger):
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.3f')
        top1 = AverageMeter('Acc@1', ':6.3f')
        top5 = AverageMeter('Acc@5', ':6.3f')
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, top1, top5],
            prefix='Eval: ')


        # switch to evaluate mode
        model.eval()
        classifier.eval()
        class_num = torch.zeros(config.num_classes).to(device)
        correct = torch.zeros(config.num_classes).to(device)

        confidence = np.array([])
        pred_class = np.array([])
        true_class = np.array([])

        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(val_loader):

                images = images.to(device)
                target = target.to(device)

                # compute output
                feat = model(images)
                output = classifier(feat)
                #loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                #losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                _, predicted = output.max(1)
                target_one_hot = F.one_hot(target, config.num_classes)
                predict_one_hot = F.one_hot(predicted, config.num_classes)
                class_num = class_num + target_one_hot.sum(dim=0).to(torch.float)
                correct = correct + (target_one_hot + predict_one_hot == 2).sum(dim=0).to(torch.float)

                prob = torch.softmax(output, dim=1)
                confidence_part, pred_class_part = torch.max(prob, dim=1)
                confidence = np.append(confidence, confidence_part.cpu().numpy())
                pred_class = np.append(pred_class, pred_class_part.cpu().numpy())
                true_class = np.append(true_class, target.cpu().numpy())

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % config.print_freq == 0:
                    progress.display(i, logger)

            acc_classes = correct / class_num
            head_acc = acc_classes[config.head_class_idx[0]:config.head_class_idx[1]].mean() * 100

            med_acc = acc_classes[config.med_class_idx[0]:config.med_class_idx[1]].mean() * 100
            tail_acc = acc_classes[config.tail_class_idx[0]:config.tail_class_idx[1]].mean() * 100
            #logger.info('* Acc@1 {top1.avg:.3f}% Acc@5 {top5.avg:.3f}% HAcc {head_acc:.3f}% MAcc {med_acc:.3f}% TAcc {tail_acc:.3f}%.'.format(top1=top1, top5=top5, head_acc=head_acc, med_acc=med_acc, tail_acc=tail_acc))
            print('* Acc@1 {top1.avg:.3f}% Acc@5 {top5.avg:.3f}% HAcc {head_acc:.3f}% MAcc {med_acc:.3f}% TAcc {tail_acc:.3f}%.'.format(top1=top1, top5=top5, head_acc=head_acc, med_acc=med_acc, tail_acc=tail_acc))

            cal = calibration(true_class, pred_class, confidence, num_bins=15)
            #logger.info('* ECE   {ece:.3f}%.'.format(ece=cal['expected_calibration_error'] * 100))
            print('* ECE   {ece:.3f}%.'.format(ece=cal['expected_calibration_error'] * 100))

        return top1.avg, cal['expected_calibration_error'] * 100,head_acc,med_acc,tail_acc


def save_checkpoint(state, is_best, model_dir):

        if is_best:
            filename = model_dir + '/model_best.pth.tar'
            torch.save(state, filename)
            #shutil.copyfile(filename, model_dir + '/model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch, config):
        """Sets the learning rate"""
        if config.cos:
            lr_min = 0
            lr_max = config.lr
            lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(epoch / config.num_epochs * 3.1415926535))
        else:
            epoch = epoch + 1
            if epoch <= 5:
                lr = config.lr * epoch / 5
            elif epoch > 180  :
                lr = config.lr * 0.01
            elif epoch > 160 :
                lr = config.lr * 0.1
            else:
                lr = config.lr

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


if __name__ == '__main__':
    main()
