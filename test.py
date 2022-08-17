from operator import index
import numpy as np
import time
import os
import ntpath
import torch.utils.data
from torch.utils.data import DataLoader
from test_options import parser
from util.data_utils import MyDataset
from model.cdnet import CDNet
from util.metrics import AverageMeter, RunningMetrics
from collections import OrderedDict
from util.utils import *
import matplotlib.pyplot as plt  
import torch.nn as nn
import cv2


opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_current_visuals(gt_value,name):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        visual_ret[name] = gt_value
        return visual_ret

def print_current_metrics(log_name, train_net, score):
    """print current acc on console; also save the losses to the disk
    Parameters:
    """
    message = '(train_net: %s) ' % str(train_net)
    for k, v in score.items():
        message += '%s: %.5f ' % (k, v)
    print(message) 
    with open(log_name, "a") as log_file:
        log_file.write('%s\n' % message) 



if __name__ == '__main__':
    n_class = opt.out_cls
    ###------------------------ load data ---------------------------###
    test_set = MyDataset(opt,opt.test1_dir, opt.test2_dir, opt.label_test)
    test_loader = DataLoader(dataset=test_set, num_workers=opt.num_workers, batch_size=opt.test_batchsize,shuffle=False)
    
    output_folder = opt.results_dir
    checkpoints_folder = os.path.join(opt.checkpoints_dir, opt.name)

    netCD = CDNet(in_c=opt.in_c, num_filters=opt.filter_num, nr_object_class=opt.out_cls).to(device, dtype=torch.float) 

    load_net = opt.load_net
    netCD.load_state_dict(torch.load(os.path.join(checkpoints_folder, load_net)))

    if torch.cuda.device_count() > 1:
        print("Use", torch.cuda.device_count(), "GPUs!")
        netCD = torch.nn.DataParallel(netCD, device_ids=range(torch.cuda.device_count())) 

    netCD.eval()

    mkdirs(output_folder)
    log_name = os.path.join(output_folder, 'test_log.txt')
    with open(log_name, "a") as log_file:
        now = time.strftime("%c")
        log_file.write('================ test acc (%s) ================\n' % now)

    running_metrics = AverageMeter()


    for image1, image2, label, index in test_loader:
        with torch.no_grad():
            image1 = image1.to(device, dtype=torch.float)
            image2 = image2.to(device, dtype=torch.float)
            label = label.to(device, dtype=torch.float)

            iter_start_time = time.time()
            segDImap, segmap1, segmap2, ds1, ds2 = netCD(image1, image2) 
            label = torch.argmax(label, 1).unsqueeze(1)

            pred = segDImap.round().long() 
            metrics = RunningMetrics(opt.out_cls)
            metrics.update(label.detach().cpu().numpy(), pred.detach().cpu().numpy())
            scores = metrics.get_cm()
            running_metrics.update(scores)

            curr_name = ntpath.basename(test_set.A_filenames[index])
            curr_name = curr_name.split(".")[0]
            visuals = get_current_visuals(pred,curr_name)

            save_visuals(visuals,output_folder,curr_name)
    score = running_metrics.get_scores()
    print_current_metrics(log_name, load_net, score)    

