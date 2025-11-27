from __future__ import division
from __future__ import print_function

# TO DO: IMPLEMENT MULTIPLE GPUs 

import torch
print("Checking if CUDA is available: ")
print(torch.cuda.is_available())
print("\n")

# Torch utils 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.dataloader as dataloader
from torch.utils.data import DataLoader, ConcatDataset
from torch.autograd import Variable
from torchvision import transforms

import os, sys, getopt 
from lib.config import config_loader
from lib.utility import timestr
import numpy as np
import time 

# import MPID core 
from mpid_data import mpid_data_multiclass
from mpid_net import mpid_net_multiclass, mpid_func_multiclass

def TrainCNN():
    '''
    Function that trains the DM-CNN model (multi-class version)
    It stores the values of the training metrics
    in a csv file that will be created in output_dir.
    The weights of the CNN will be stored in weights_dir

    The input files, output directories and settings for this 
    function are given in a config file
    '''

    print("Reading config file...\n")
    BASE_PATH = os.path.realpath(__file__)
    BASE_PATH = os.path.dirname(BASE_PATH)
    CFG = os.path.join(BASE_PATH,"../cfg","training_config_multiclass.cfg")
    cfg  = config_loader(CFG)

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPUID

    title = cfg.name
    output_dir = cfg.output_directory
    train_files = cfg.input_train.split(',')
    test_files = cfg.input_test.split(',')
    weights_dir = cfg.weights_directory 
    SEED = cfg.seed_number 

    print("Inputs given: ")
    print("Training files: ", train_files)
    print("Test files: ", test_files)
    print("Output directory: ", output_dir)
    print("Weights directory: ", weights_dir)
    print("Seed?: ", cfg.seed)
    print("\n")

    fout = open(output_dir + 'DM-CNN_training_metrics_{}_{}.csv'.format(timestr(), title), 'w')
    fout.write('train_accu,test_accu,train_loss,test_loss,epoch,step\n')

    CNN_weights = weights_dir + "DM-CNN_model_{}_epoch_{}_batch_id_{}_labels_{}_title_{}_step_{}.pwf"

    cuda = torch.cuda.is_available()
    if(cfg.seed and cuda):
        print("Using seed number: ", SEED)
        torch.cuda.manual_seed(SEED)
    elif(cfg.seed and not cuda):
        print("Using seed number: ", SEED)
        torch.manual_seed(SEED)
    else:
        print("A seed has not been defined...\n")

    print("There are {} GPUs available".format(torch.cuda.device_count()))
    train_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Training data
    train_datasets = [
        mpid_data_multiclass.MPID_Dataset(f, "image2d_image2d_binary_tree", train_device, plane=0, augment=False)
        for f in train_files
    ]
    train_data = ConcatDataset(train_datasets)
    train_loader = DataLoader(dataset=train_data, batch_size=cfg.batch_size_train, shuffle=True)
    labels = cfg.num_class

    # Test data
    test_datasets = [
        mpid_data_multiclass.MPID_Dataset(f, "image2d_image2d_binary_tree", train_device, plane=0)
        for f in test_files
    ]
    test_data = ConcatDataset(test_datasets)
    test_loader = DataLoader(dataset=test_data, batch_size=cfg.batch_size_test, shuffle=True)

    # Import the CNN model 
    mpid = mpid_net_multiclass.MPID(dropout=cfg.drop_out, num_classes=cfg.num_class)
    mpid.cuda()

    loss_fn = nn.CrossEntropyLoss()
    optimizer  = optim.Adam(mpid.parameters(), lr=cfg.learning_rate)
    train_step = mpid_func_multiclass.make_train_step(mpid, loss_fn, optimizer)
    test_step  = mpid_func_multiclass.make_test_step(mpid, test_loader, loss_fn, optimizer)

    print("Training with {} images".format(len(train_loader.dataset)))

    train_losses = []
    train_accuracies =[]
    test_losses = []
    test_accuracies =[]

    EPOCHS = cfg.EPOCHS
    print("Start DM-CNN training...")

    step=0
    init = time.time()
    for epoch in range(EPOCHS):
        print("\n@{}th epoch...".format(epoch))
        for batch_idx, (x_batch, y_batch, info_batch, nevents_batch) in enumerate(train_loader):
            print("\n@{}th epoch, @ batch_id {}".format(epoch, batch_idx))
            x_batch = x_batch.to(train_device).view((-1,1,512,512))
            y_batch = y_batch.to(train_device)
            loss = train_step(x_batch, y_batch)
            train_losses.append(loss)
            print('\r Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                EPOCHS-1,
                batch_idx * len(x_batch), 
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader), 
                loss), 
                end='')
            if (batch_idx % cfg.test_every_step == 1 and cfg.run_test):
                if (cfg.save_weights and epoch >= 3 and epoch <= 6):
                    torch.save(mpid.state_dict(), CNN_weights.format(timestr(), epoch, batch_idx,labels, title, step))

                print("Start eval on test sample.......@step..{}..@epoch..{}..@batch..{}".format(step,epoch, batch_idx))
                test_accuracy = mpid_func_multiclass.validation(mpid, test_loader, cfg.batch_size_test, train_device, event_nums=cfg.test_events_nums)
                print("Test Accuracy {}".format(test_accuracy))
                print("Start eval on training sample...@epoch..{}.@batch..{}".format(epoch, batch_idx))
                train_accuracy = mpid_func_multiclass.validation(mpid, train_loader, cfg.batch_size_train, train_device, event_nums=cfg.test_events_nums)
                print("Train Accuracy {}".format(train_accuracy))
                test_loss= test_step(test_loader, train_device)
                print("Test Loss {}".format(test_loss))
                fout.write("%f,%f,%f,%f,%f,%f\n" % (train_accuracy, test_accuracy, loss, test_loss, epoch, step))
            step+=1
    fout.close()
    end = time.time()
    print("\nTotal training time: {:0.4f} seconds".format(end-init))
    return 0 

if __name__ == '__main__':
    TrainCNN()
