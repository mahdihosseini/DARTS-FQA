import os
import sys
import time
import glob
import numpy as np
import pandas as pd
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from copy import deepcopy
from numpy import linalg as LA
import matplotlib.pyplot as plt

from utils import _data_transforms_focuspath
from plcc_loss import PLCCLoss
from torch.autograd import Variable
from dartsfqa_search import Network
from architect import Architect
from analyze import Analyzer
from dataset import FocusDataset
from adas import Adas
from adas.metrics import Metrics
from adas.adaptive_stop import StopChecker

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=50, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.175, help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--epochs', type=int, default=60, help='num of training epochs')
    parser.add_argument('--init_channels', type=int, default=8, help='num of init channels')
    parser.add_argument('--layers', type=int, default=3, help='total number of layers')
    parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--drop_path_prob', type=float, default=0.0, help='drop path probability')
    # Change to FOCUSPATH or DEEPFOCUS depending on the dataset.
    parser.add_argument('--save', type=str, default='BIOIMAGE_exp', help='experiment name, options: BIOIMAGE_exp, FOCUSPATH_exp, DEEPFOCUS_exp')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
    parser.add_argument('--train_portion', type=float, default=0.50, help='portion of training data')
    parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
    parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
    # BN
    parser.add_argument('--learnable_bn', action='store_true', default=False, help='learnable parameters in batch normalization')
    # Gumbel-softmax
    parser.add_argument('--gumbel', action='store_true', default=False, help='use or not Gumbel-softmax trick')
    parser.add_argument('--tau_max', type=float, default=10.0, help='initial tau')
    parser.add_argument('--tau_min', type=float, default=1.0, help='minimum tau')
    # Dataset
    parser.add_argument('--dataset', type=str, default='FocusPath', help='choose dataset, options: BioImage, DeepFocus, FocusPath')
    # Adas optimizer
    parser.add_argument('--adas', action='store_true', default=False, help='whether or not to use adas optimizer')
    parser.add_argument('--scheduler_beta', type=float, default=0.97, help='beta for lr scheduler')
    parser.add_argument('--step_size', type=int, default=None, help='step_size for dropping zeta')
    parser.add_argument('--gamma', type=float, default=0.5, help='zeta dropping rate in Adas')
    # Save file name
    parser.add_argument('--file_name', type=str, default='focuspath', help='metrics and weights data file name')
    # Hessian
    parser.add_argument('--compute_hessian', action='store_true', default=False, help='compute or not Hessian')
    parser.add_argument('--report_freq_hessian', type=int, default=2, help='frequency to report Hessian')
    # Local stopping criterion
    parser.add_argument('--adaptive_stop', action='store_true', default=False, help='local stopping criterion')
    parser.add_argument('--as_start_epoch', type=int, default=10, help='start epoch for local stopping criterion')
    parser.add_argument('--num_normal_cell_stop', type=int, default=2, help='param for local stopping criterion')
    parser.add_argument('--num_reduce_cell_stop', type=int, default=1, help='param for local stopping criterion')
    # Global stopping criterion
    parser.add_argument('--global_stop', action='store_true', default=False, help='global stopping criterion')
    parser.add_argument('--gs_factor', type=float, default=1.3, help='factor for global stopping criterion')
    parser.add_argument('--gs_start_epoch', type=int, default=20, help='start epoch for global stopping criterion')
    parser.add_argument('--gs_delta', type=int, default=2, help='delta for global stopping criterion')
    # Intermediate nodes in a cell
    parser.add_argument('--node', type=int, default=2, help='number of nodes in a cell')
    # Path Information
    parser.add_argument('--result_path', type=str, default='./search_results', metavar='PATH', help='Path for save loss plots.')
    parser.add_argument('--dataset_path', type=str, default='./data', metavar='PATH', help='Path at which the dataset is stored.')
    parser.add_argument('--csv_path', type=str, default='./data', metavar='PATH', help='Path at which the CSV is stored.')


    args = parser.parse_args()

    args.save = 'search-{}-{}-{}'.format(args.save, args.file_name, time.strftime("%Y%m%d-%H%M%S"))
    save_folder_parent = args.result_path
    if not os.path.exists(save_folder_parent):
        os.makedirs(save_folder_parent)
    save_folder = save_folder_parent + "%flr_%s" % (args.learning_rate, args.dataset)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    args.save = os.path.join(save_folder, args.save)
    utils.create_exp_dir(args.save, scripts_to_save=None)
    # utils.create_exp_dir(args.save, scripts_to_save=glob.glob(*.py))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    return args

def main(args):
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    torch.cuda.empty_cache()
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    if args.dataset == 'cifar100':
        n_classes = 100
        data_folder = 'cifar-100-python'
    elif args.dataset.lower() == 'focuspath':
        n_classes = 15
    elif args.dataset.lower() == 'deepfocus':
        n_classes = 6
    elif args.dataset.lower() == 'bioimage':
        n_classes = 11
    else:
        n_classes = 10
        data_folder = 'cifar-10-batches-py'

    # Want to use the PLCC loss, not Cross Entropy.
    # criterion = nn.CrossEntropyLoss()
    # criterion = criterion.cuda()
    criterion = PLCCLoss()
    criterion = criterion.cuda()
    model = Network(args.init_channels, n_classes, args.layers, criterion, args.dataset, args.batch_size, learnable_bn=args.learnable_bn, steps=args.node)
    model = model.cuda()

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    logging.info("num of params = %d", num_param)

    ################################################################################
    # AdaS: optimizer and scheduler
    if args.adas:
        optimizer = Adas(params=list(model.parameters()),
                         lr=args.learning_rate,
                         beta=args.scheduler_beta,
                         step_size=args.step_size,
                         gamma=args.gamma,
                         momentum=args.momentum,
                         weight_decay=args.weight_decay)

    ################################################################################
    # original DARTS: SGD optimizer with cosine_annealing lr scheduler
    # else:
    #     optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    #     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_interval, gamma=args.decay_ratio)
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay)

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, float(args.epochs), eta_min=args.learning_rate_min)
    ################################################################################

    if args.dataset == 'cifar100':
        train_transform, valid_transform = utils._data_transforms_cifar100(args)
        train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
    elif args.dataset.lower() == 'focuspath':
        train_transform, valid_transform = utils._data_transforms_focuspath(args)
        train_csv = args.csv_path
        trainset = args.dataset_path
    elif args.dataset.lower() == 'bioimage':
        train_transform, valid_transform = utils._data_transforms_bioimage(args)
        train_csv = args.csv_path
        trainset = args.dataset_path
    elif args.dataset.lower() == 'deepfocus':
        train_transform, valid_transformm = utils._data_transforms_deepfocus(args)
        train_csv = args.csv_path
        trainset = args.dataset_path
    else:
        train_transform, valid_transform = utils._data_transforms_cifar10(args)
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

    if args.dataset == 'cifar100' or args.dataset == 'cifar10':
        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(args.train_portion * num_train))

        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
            pin_memory=True, num_workers=0)

        valid_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
            pin_memory=True, num_workers=0)
    elif args.dataset.lower() == 'focuspath':
        train_data = FocusDataset(csv_file=train_csv, root_dir=trainset, transform=train_transform, dataset=args.dataset)
        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(args.train_portion * num_train))
        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
            pin_memory=True, num_workers=0, drop_last=True)
        valid_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
            pin_memory=True, num_workers=0, drop_last=True)
    elif args.dataset.lower() == 'bioimage':
        train_data = FocusDataset(csv_file=train_csv, root_dir=trainset, transform=train_transform, dataset=args.dataset)
        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(args.train_portion * num_train))
        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
            pin_memory=True, num_workers=1, drop_last=True)
        valid_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
            pin_memory=True, num_workers=1, drop_last=True)
    elif args.dataset.lower() == 'deepfocus':
        train_data = FocusDataset(csv_file=train_csv, root_dir=trainset, transform=train_transform, dataset=args.dataset)
        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(args.train_portion * num_train))
        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
            pin_memory=True, num_workers=0, drop_last=True)
        valid_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
            pin_memory=True, num_workers=0, drop_last=True)

    architect = Architect(model, args)
    """Hessian"""
    analyser = Analyzer(model, args)
    """adaptive stopping based on knowledge gains"""
    stop_checker = StopChecker(args)
    """global stopping based on eigen values"""
    ev_tracker = utils.EVLocalAvg(args)

    if not args.adas:
        # adas has already called this
        metrics = Metrics(params=list(model.parameters()))

    performance_statistics = {}
    arch_statistics = {}
    genotype_statistics = {}

    save_folder = args.result_path + "%flr_%s" % (args.learning_rate, args.dataset)
    metrics_path = save_folder + 'metrics_stat_' + args.file_name + '.xlsx'
    weights_path = save_folder + 'weights_stat_' + args.file_name + '.xlsx'
    genotypes_path = save_folder + 'genotypes_stat_' + args.file_name + '.xlsx'

    edge_num = np.sum([2+i for i in range(args.node)])
    normal_edge_stop_epoch = np.zeros(edge_num)
    reduce_edge_stop_epoch = np.zeros(edge_num)
    local_stop_epoch = None
    errors_dict = {'train_acc': [], 'train_loss': [], 'valid_acc': [], 'valid_loss': []}


    for epoch in range(args.epochs):
        if args.adas:
            lr = optimizer.lr_vector
        else:
            logging.info('epoch %d lr %e', epoch, args.learning_rate)

        genotype = model.genotype()
        logging.info('epoch: %d', epoch)
        logging.info('genotype = %s', genotype)

        # Training for DARTS-FQA
        train_obj = train(args, epoch, train_queue, valid_queue,
                          model, architect, criterion,
                          optimizer, args.learning_rate,
                          analyser, ev_tracker)

        print('\n')
        logging.info('train_loss %f', train_obj)

        # Validation for DARTS-FQA
        valid_obj = infer(args, valid_queue, model, criterion)
        print('\n')
        logging.info('valid_loss %f', valid_obj)

        # update the errors dictionary
        errors_dict['train_loss'].append(train_obj)
        errors_dict['valid_loss'].append(valid_obj)

        losses = [train_obj, valid_obj]

        # update network io metrics (knowledge gain, condition mapping, etc)
        if args.adas:
            # AdaS: update learning rates
            optimizer.epoch_step(epoch)
            io_metrics = optimizer.KG
            lr_metrics = optimizer.velocity
        else:
            metrics()
            io_metrics = metrics.KG(epoch)
            lr_metrics = None
        # weights
        weights_normal = F.softmax(model.alphas_normal, dim=-1).detach().cpu().numpy()
        weights_reduce = F.softmax(model.alphas_reduce, dim=-1).detach().cpu().numpy()

        # write data to excel files
        loss_list = write_data(args, epoch, io_metrics, lr_metrics, weights_normal, weights_reduce, genotype,
                                  performance_statistics, arch_statistics, genotype_statistics,
                                  metrics_path, weights_path, genotypes_path, losses, loss_list)


        # save model parameters
        utils.save(model, os.path.join(args.save, 'weights.pt'))

        # adaptive stopping criterion (local, based on knowledge gain)
        if args.adaptive_stop and epoch >= args.as_start_epoch:
            # apply local stopping criterion
            stop_checker.local_stop(optimizer.metrics, epoch)
            # freeze some edges based on their knowledge gains
            iteration_p = 0
            for p in model.parameters():
                if ~optimizer.metrics.layers_index_todo[iteration_p]:
                    p.requires_grad = False
                    p.grad = None
                iteration_p += 1

            for i in range(edge_num):
                if stop_checker.normal_edge_index_stop[i] & (normal_edge_stop_epoch[i] == 0):
                    normal_edge_stop_epoch[i] = epoch
                if stop_checker.reduce_edge_index_stop[i] & (reduce_edge_stop_epoch[i] == 0):
                    reduce_edge_stop_epoch[i] = epoch
            logging.info(
                'Epoch: %d, normal edge stop epoch: %s',
                epoch, str(normal_edge_stop_epoch)
            )
            logging.info(
                'Epoch: %d, reduce edge stop epoch: %s',
                epoch, str(reduce_edge_stop_epoch)
            )

            if min(normal_edge_stop_epoch) > 0 and min(reduce_edge_stop_epoch) > 0:
                if local_stop_epoch is None:
                    logging.info(
                        'All edges are frozen at epoch: %d', epoch
                    )
                    local_stop_epoch = epoch + 2
                # if all edges are frozen, we should
                # stop searching the whole net after 2 epochs
                elif epoch == local_stop_epoch:
                    logging.info(
                        'Based on the local criterion, decide to stop the search at epoch %d', epoch
                    )
                    # FocusPath
                    logging.info(
                        'Validation loss at stop epoch: %f' , valid_obj
                    )
                    # logging.info(
                    #     'Validation accuracy at stop epoch: %f', valid_acc
                    # )
                    logging.info(
                        'Genotype at stop epoch: %s', genotype
                    )
                    break
                else:
                    logging.info(
                        'Waiting for the searching to stop in %d epochs', local_stop_epoch - epoch
                    )

        # global stop criterion (global, based on Hessian matrix)
        if args.global_stop:
            if ev_tracker.stop_search:
                # set the following to the values they had at stop_epoch
                errors_dict['valid_acc'] = errors_dict['valid_acc'][:ev_tracker.stop_epoch + 1]
                stop_genotype = ev_tracker.stop_genotype
                stop_valid_acc = errors_dict['valid_acc'][ev_tracker.stop_epoch]
                logging.info(
                    'Based on the global criterion, decide to stop the search at epoch %d (Current epoch: %d)',
                    ev_tracker.stop_epoch, epoch
                )
                logging.info(
                    'Validation accuracy at stop epoch: %f', stop_valid_acc
                )
                logging.info(
                    'Genotype at stop epoch: %s', stop_genotype
                )
                break


def train(args, epoch, train_queue, valid_queue, model, architect, criterion, optimizer, lr, analyser, ev_tracker):
    objs = utils.AvgrageMeter()
    train_loss = 0
    if args.adas:
        layers_todo = optimizer.metrics.layers_index_todo
    else:
        layers_todo = None

    for step, sample_batched in enumerate(train_queue, 0):
        input, target = sample_batched['image'], sample_batched['score']
        # one mini-batch
        print('\rtrain mini batch {:03d}'.format(step), end=' ')
        model.train()

        if args.gumbel:
            model.set_tau(args.tau_max - epoch * 1.0 / args.epochs * (args.tau_max - args.tau_min))

        input = Variable(input, requires_grad=False).cuda()
        target = Variable(target.float(), requires_grad=False).cuda(non_blocking=True)

        # get a random minibatch from the search queue with replacement
        sample_batched_search = next(iter(valid_queue))
        input_search, target_search = sample_batched_search['image'], sample_batched_search['score']
        input_search = Variable(input_search, requires_grad=False).cuda()
        target_search = Variable(target_search.float(), requires_grad=False).cuda(non_blocking=True)

        # logging.info('update arch...')
        architect.step(input, target, input_search, target_search, lr, layers_todo, optimizer, unrolled=args.unrolled)

        # logging.info('update weights...')
        optimizer.zero_grad()

        logits = model.forward(input, gumbel=args.gumbel)
        logits_avg = logits.view(args.batch_size, 1).mean(1)
        loss = criterion(logits_avg, target)

        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        train_loss += loss.item()
        print_train_loss = train_loss / (step + 1)

        if step % args.report_freq == 0:
            print('\n')
            logging.info('train %03d %f', step, print_train_loss)
            # logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    if args.compute_hessian:
        if (epoch % args.report_freq_hessian == 0) or (epoch == (args.epochs - 1)):
            _data_loader = deepcopy(train_queue)
            sample_batched = next(iter(_data_loader))

            input, target = sample_batched['image'], sample_batched['score']

            input = Variable(input, requires_grad=False).cuda()
            target = Variable(target.float(), requires_grad=False).cuda(non_blocking=True)

            # get gradient information
            # param_grads = [p.grad for p in model.parameters() if p.grad is not None]
            # param_grads = torch.cat([x.view(-1) for x in param_grads])
            # param_grads = param_grads.cpu().data.numpy()
            # grad_norm = np.linalg.norm(param_grads)

            # gradient_vector = torch.cat([x.view(-1) for x in gradient_vector])
            # grad_norm = LA.norm(gradient_vector.cpu())
            # logging.info('\nCurrent grad norm based on Train Dataset: %.4f',
            #             grad_norm)
            # logging.info('Compute Hessian start')
            H = analyser.compute_Hw(input, target, input_search, target_search,
                                    lr, layers_todo, optimizer, unrolled=False)
            # g = analyser.compute_dw(input, target, input_search, target_search,
            #                         lr, layers_todo, optimizer, unrolled=False)
            # g = torch.cat([x.view(-1) for x in g])

            del _data_loader
            hessian_file = "../save_data/hessian_{0}_epoch_{1}".format(args.file_name, epoch)
            np.save(hessian_file, H.cpu().data.numpy())

            # early stopping
            ev = max(LA.eigvals(H.cpu().data.numpy()))

            ev_tracker.update(epoch, ev, model.genotype())
            if args.global_stop and epoch != (args.epochs - 1):
                ev_tracker.early_stop(epoch,
                                      factor=args.gs_factor,
                                      es_start_epoch=args.gs_start_epoch,
                                      delta=args.gs_delta)

    return print_train_loss
    # return top1.avg, objs.avg


def infer(args, valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    # top1 = utils.AvgrageMeter()
    # top5 = utils.AvgrageMeter()
    model.eval()
    valid_loss = 0

    with torch.no_grad():
        for step, sample_batched in enumerate(valid_queue):
            print('\rinfer mini batch {:03d}'.format(step), end=' ')
            input, target = sample_batched['image'], sample_batched['score']

            input = Variable(input).cuda()
            target = Variable(target.float()).cuda(non_blocking=True)

            logits = model(input)
            logits_avg = logits.view(args.batch_size, 1).mean(1)
            loss = criterion(logits_avg, target)

            valid_loss += loss.item()
            print_valid_loss = valid_loss / (step + 1)

            if step % args.report_freq == 0:
                print('\n')
                logging.info('valid %03d %f', step, print_valid_loss)

    return print_valid_loss


def write_data(args, epoch, net_metrics, lr_metrics, weights_normal, weights_reduce, genotype,
               perform_stat, arch_stat, genotype_stat, metrics_path, weights_path, genotypes_path, losses, loss_list):
    # io metrics
    perform_stat['S_epoch_{}'.format(epoch)] = net_metrics
    if args.adas:
        # lr metrics
        perform_stat['learning_rate_epoch_{}'.format(epoch)] = lr_metrics
        learning_rate_print = np.array(perform_stat['learning_rate_epoch_{}'.format(epoch)])
        learning_rate_print = np.mean(learning_rate_print)

    loss_list[0, epoch] = losses[0]
    loss_list[1, epoch] = losses[1]

    result_path = args.result_path
    valid_summary_file = "%flr_valid_loss.txt" % args.learning_rate
    train_summary_file = "%flr_train_loss.txt" % args.learning_rate
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    valid_result = os.path.join(result_path, valid_summary_file)
    train_result = os.path.join(result_path, train_summary_file)


    # genotype
    if epoch % 5 == 0 or epoch == args.epochs - 1:
        genotype_stat['epoch_{}'.format(epoch)] = [genotype]
        genotypes_df = pd.DataFrame(data=genotype_stat)
        genotypes_df.to_excel(metrics_path)

        valid_result_file = open(valid_result, 'a')
        format_out = '(E:%d) [loss = %.4f, lr = %.6e]'
        print_out = format_out % (epoch, losses[1], args.learning_rate)
        valid_result_file.write(print_out + '\n')
        valid_result_file.write(str(genotype) + '\n')
        print(str(genotype))
        valid_result_file.close()

        train_result_file = open(train_result, 'a')
        format_out = '(E:%d) [loss = %.4f, lr = %.6e]'
        print_out = format_out % (epoch, losses[0], args.learning_rate)
        train_result_file.write(print_out + '\n')
        train_result_file.write(str(genotype) + '\n')
        print(str(genotype))
        train_result_file.close()

    plt.ion()
    plt.figure()
    plt.plot(range(epoch), loss_list[0, 0:epoch], label='Train Loss')
    plt.plot(range(epoch), loss_list[1, 0:epoch], label='Validation Loss')
    plt.legend()
    plt.title('Architecture Search Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (PLCC)')
    loss_plot = "loss_%flr.png" % args.learning_rate
    loss_plot_file = os.path.join(result_path, loss_plot)
    plt.savefig(loss_plot_file)
    plt.show(block=False)
    plt.pause(3)
    print("Loss plot finished for epoch %d" % epoch)
    plt.ioff()
    plt.close('all')


    # write metrics data to xls file
    metrics_df = pd.DataFrame(data=perform_stat)
    metrics_df.to_excel(metrics_path)

    # weights
    # normal
    arch_stat['normal_none_epoch{}'.format(epoch)] = weights_normal[:, 0]
    arch_stat['normal_max_epoch{}'.format(epoch)] = weights_normal[:, 1]
    arch_stat['normal_avg_epoch{}'.format(epoch)] = weights_normal[:, 2]
    arch_stat['normal_skip_epoch{}'.format(epoch)] = weights_normal[:, 3]
    arch_stat['normal_sep_3_epoch{}'.format(epoch)] = weights_normal[:, 4]
    arch_stat['normal_sep_5_epoch{}'.format(epoch)] = weights_normal[:, 5]
    arch_stat['normal_dil_3_epoch{}'.format(epoch)] = weights_normal[:, 6]
    arch_stat['normal_dil_5_epoch{}'.format(epoch)] = weights_normal[:, 7]
    # reduce
    arch_stat['reduce_none_epoch{}'.format(epoch)] = weights_reduce[:, 0]
    arch_stat['reduce_max_epoch{}'.format(epoch)] = weights_reduce[:, 1]
    arch_stat['reduce_avg_epoch{}'.format(epoch)] = weights_reduce[:, 2]
    arch_stat['reduce_skip_epoch{}'.format(epoch)] = weights_reduce[:, 3]
    arch_stat['reduce_sep_3_epoch{}'.format(epoch)] = weights_reduce[:, 4]
    arch_stat['reduce_sep_5_epoch{}'.format(epoch)] = weights_reduce[:, 5]
    arch_stat['reduce_dil_3_epoch{}'.format(epoch)] = weights_reduce[:, 6]
    arch_stat['reduce_dil_5_epoch{}'.format(epoch)] = weights_reduce[:, 7]
    # write weights data to xls file
    weights_df = pd.DataFrame(data=arch_stat)
    weights_df.to_excel(weights_path)

    return loss_list


if __name__ == '__main__':
    args = parse_config()
    main(args)
