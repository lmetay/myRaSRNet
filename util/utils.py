
"""This module contains simple helper functions """
import torch
import os
import ntpath
import numpy as np
from PIL import Image

def get_scheduler(optimizer, opt, iter_num):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions.
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    from torch.optim import lr_scheduler
    from util.lr_scheduler import Poly, OneCycle

    if opt.lr_policy == 'linear': 
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1) 
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step': 
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=5e-6)
    elif opt.lr_policy == 'warmup':
        scheduler = Poly(optimizer, num_epochs=opt.niter+opt.niter_decay, iters_per_epoch=iter_num)
    elif opt.lr_policy == 'onecycle':
        scheduler = OneCycle(optimizer, num_epochs=opt.niter+opt.niter_decay, iters_per_epoch=iter_num)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def get_current_losses(self):
    """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
    from collections import OrderedDict
    errors_ret = OrderedDict()
    for name in self.loss_names:
        if isinstance(name, str):
            errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
    return errors_ret


def save_networks(self, epoch):
    """Save all the networks to the disk.

    Parameters:
        epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
    """
    for name in self.model_names:
        if isinstance(name, str):
            save_filename = '%s_net_%s.pth' % (epoch, name)
            save_path = os.path.join(self.save_dir, save_filename)
            net = getattr(self, 'net' + name)

            if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                # torch.save(net.module.cpu().state_dict(), save_path)
                torch.save(net.cpu().state_dict(), save_path)
                net.cuda(self.gpu_ids[0])
            else:
                torch.save(net.cpu().state_dict(), save_path)


def print_current_acc(log_name, epoch, score):  
    """print current acc on console; also save the losses to the disk
    Parameters:
    """
    message = '(epoch: %d) ' % epoch
    for k, v in score.items():
        message += '%s: %.5f ' % (k, v)
    print(message)  # print the message
    with open(log_name, "a") as log_file:
        log_file.write('%s\n' % message)  # save the message


def cache_model(model, str, opt):
    save_filename = '%s_net.pth' % (str) 
    save_path = os.path.join(opt.checkpoints_dir, opt.name, save_filename)
    if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
        torch.save(model.cpu().state_dict(), save_path)
        model.cuda()
    else:
        torch.save(model.state_dict(), save_path)


def update_learning_rate(schedulers, opt, optimizers):
        """Update learning rates for all the networks; called at the end of every epoch"""
        if opt.lr_policy == 'plateau':
            schedulers.step(int(0))
        else:
            schedulers.step()

        lr = optimizers.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)


def print_options(opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def save_visuals(visuals,img_dir,name):
    """
    """
    name = ntpath.basename(name)
    name = name.split(".")[0]
    print(name)
    # save images to the disk
    for label, image in visuals.items():
        image_numpy = tensor2im(image) #image int64 [B,1,H,W]
        img_path = os.path.join(img_dir, '%s.png' % (name))
        save_image(image_numpy, img_path)


def tensor2im(input_image, imtype=np.uint8, normalize=True):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))

        image_numpy = np.transpose(image_numpy, (1, 2, 0))
        if normalize:
            image_numpy = (image_numpy + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling

    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def draw_features(width,height,x):
    import matplotlib.pyplot as plt
    import numpy as np
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    for i in range(width*height):
        plt.subplot(height,width, i + 1)
        plt.axis('off')
        img = x[0, i, :, :]   #x[:,:,i]
        pmin = np.min(img)
        pmax = np.max(img)
        img = (img - pmin) / (pmax - pmin + 0.000001)
        plt.imshow(img, cmap='gray')
    fig.clf()