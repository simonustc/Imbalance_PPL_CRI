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
    parser = argparse.ArgumentParser(description='MiSLAS training (Stage-1)')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='./config/ina2018/ina2018_CE.yaml',
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

        if config.dataset == 'cifar10' or config.dataset == 'cifar10_CRI':
            dataset = CIFAR10_LT(root=config.data_path, imb_ratio=config.imb_ratio,imb_type=config.imb_type,imb_factor=config.imb_factor,
                                 batch_size=config.batch_size, num_works=config.workers)

        elif config.dataset == 'cifar100' or config.dataset == 'cifar100_CRI':
            dataset = CIFAR100_LT(root=config.data_path, imb_ratio=config.imb_ratio,imb_type=config.imb_type,imb_factor=config.imb_factor,
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

        optimizer = torch.optim.SGD([{"params": model.parameters()},
                                        {"params": classifier.parameters()}], config.lr,
                                        momentum=config.momentum,
                                        weight_decay=config.weight_decay)

        best_acc1=0
        ii=0

        for epoch in range(config.num_epochs):

            adjust_learning_rate(optimizer, epoch, config)


            train(train_loader,device, num_class_list,model, classifier, criterion, epoch,optimizer, config, logger)

            acc1, ece,head_acc,med_acc,tail_acc = validate(val_loader,device, model, classifier, criterion, epoch,config, logger)


            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            if is_best:
                its_ece = ece
            logger.info('Best Prec@1: %.3f%% ECE: %.3f%%  epoch:%.3f\n' % (best_acc1, its_ece,epoch))


            save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict_model': model.state_dict(),
                        'state_dict_classifier': classifier.state_dict(),
                        'best_acc1': best_acc1,
                        'its_ece': its_ece,
                    }, is_best, model_dir)



        be_dir = Path(config.output_dir) / Path(config.iter)
        best_dir = os.path.join(be_dir,str(best_acc1))

        if not os.path.exists(best_dir):
            os.makedirs(best_dir)

        best.append(best_acc1)
        a = max(best)
        b = best.index(a)

        txt_dir = os.path.join(be_dir, 'best.txt')
        with open(txt_dir, 'a', encoding='utf-8') as f:
            f.write("best_acc:   " + str(best_acc1) + '\n')
            f.write("max_best:    " + str(a) + '\n')
            f.write("max_index   " + str(b) + '\n')

        logging.shutdown()



def train(train_loader, device,num_class_list,model, classifier, criterion,epoch, optimizer, config, logger):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.3f')
        top1 = AverageMeter('Acc@1', ':6.3f')
        top5 = AverageMeter('Acc@5', ':6.3f')
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch))
        num_class_list1 = torch.FloatTensor(num_class_list)


        model.train()
        classifier.train()

        train_loader.dataset.update(epoch)

        print("lr rate is:",optimizer.param_groups[0]['lr'])


        training_data_num = len(train_loader.dataset)
        end_steps = int(training_data_num / train_loader.batch_size)
        criterion.reset_epoch(epoch)

        end = time.time()

        for i, (images, target) in enumerate(train_loader):
            if i > end_steps:
                break

            # measure data loading time
            data_time.update(time.time() - end)

            if torch.cuda.is_available():
                images = images.to(device)
                target = target.to(device)


            if config.mixup=='ppmix':
                l = np.random.beta(config.alpha, config.alpha)
                idx = torch.randperm(images.size(0)).to(device)
                image_a, image_b = images, images[idx]
                label_a, label_b = target, target[idx]
                mixed_image = l * image_a + (1 - l) * image_b
                mixed_image = mixed_image.to(device)
                feat = model(mixed_image)
                output = classifier(feat)

                # what remix does
                l_list = torch.empty(images.shape[0]).fill_(l).float().to(device)
                #n_i, n_j = num_class_list1[label_a], num_class_list1[label_b].float()

                n_i = num_class_list1[label_a].float().to(device)
                n_j = num_class_list1[label_b].float().to(device)

                if l < 0.5:
                    if epoch < config.ppmix_epoch_min:
                        l_list[n_i / n_j >= 3] = l
                    if epoch >= config.ppmix_epoch_min and epoch <= config.ppmix_epoch_max:
                        l_list[n_i / n_j >= 3] = l*(1-((epoch-config.ppmix_epoch_min)/(config.ppmix_epoch_max-config.ppmix_epoch_min))**config.ppm_alpha)

                    if epoch > config.ppmix_epoch_max:
                        l_list[n_i / n_j >= 3] = 0

                if 1 - l < 0.5:
                    if epoch < config.ppmix_epoch_min:
                        l_list[(n_i * 3) / n_j <= 1] = l
                    if epoch >= config.ppmix_epoch_min and epoch <= config.ppmix_epoch_max:
                        l_list[(n_i * 3) / n_j <= 1] = l*(1-((epoch-config.ppmix_epoch_min)/(config.ppmix_epoch_max-config.ppmix_epoch_min))**config.ppm_alpha)+((epoch-config.ppmix_epoch_min)/(config.ppmix_epoch_max-config.ppmix_epoch_min))**config.ppm_alpha
                    if epoch > config.ppmix_epoch_max:
                        l_list[(n_i * 3) / n_j <= 1] = 1

                # print(l_list)

                label_a = label_a.to(device)
                label_b = label_b.to(device)
                loss = l_list * criterion(output, label_a, epoch) + (1 - l_list) * criterion(output, label_b, epoch)
                loss = loss.mean()

            else:

                feat = model(images)
                output = classifier(feat)

                loss = criterion(output, target,epoch)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.print_freq == 0:
                progress.display(i, logger)



def validate(val_loader,device,model, classifier, criterion, epoch,config, logger):
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
                loss = criterion(output, target,epoch)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
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
