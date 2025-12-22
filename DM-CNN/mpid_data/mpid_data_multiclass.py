import numpy as np
from larcv import larcv
import random
import os
import ROOT
from ROOT import TChain
import torch
from torch.utils.data import Dataset

def image_modify(img):
    img_mod = np.where(img < 10, 0, img)
    img_mod = np.where(img > 500, 500, img_mod)
    return img_mod

class MPID_Dataset(Dataset):
    def __init__(self, input_file, image_tree, device, plane=0, augment=False, verbose=False):
        self.plane = plane
        self.augment = augment
        self.verbose = verbose
        self.input_file = input_file
        
        self.particle_image_chain = TChain(image_tree)
        self.particle_image_chain.AddFile(input_file)
        self.device = device if device else "cpu"
        
        # Assign class label based on filename - 3 CLASSES ONLY
        if "ncpi0_corsika" in input_file:        self.class_label = 0
        elif "cosmics_corsika" in input_file:    self.class_label = 1
        elif "dm_signal_only" in input_file:     self.class_label = 2
        else:
            self.class_label = 0  # Default for inference on unknown files
    
    def __getitem__(self, ENTRY):
        self.particle_image_chain.GetEntry(ENTRY)
        img_cpp = self.particle_image_chain.image2d_image2d_binary_branch
        img_np = larcv.as_ndarray(img_cpp.as_vector()[self.plane])
        
        img_np = image_modify(img_np)
        
        if self.augment:
            if random.randint(0, 1):
                img_np = np.fliplr(img_np)
            if random.randint(0, 1):
                img_np = img_np.transpose(1, 0)
        
        img_tensor = torch.from_numpy(img_np.copy()).float()
        img_tensor = img_tensor.clone().detach()
        
        label_tensor = torch.tensor(self.class_label).long()
        
        event_info = [img_cpp.run(), img_cpp.subrun(), img_cpp.event()]
        nevents = self.particle_image_chain.GetEntries()
        
        return (img_tensor, label_tensor, event_info, nevents)
    
    def __len__(self):
        return self.particle_image_chain.GetEntries()
