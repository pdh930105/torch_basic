import argparse
import datetime
import logging
import os
import time
import traceback
import sys
import copy

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import timm

from dataset.dataset import create_loader
from utils import Logger
from optimizer import get_optimizer_scheduler
from options import Option
from trainer import Trainer

class ExperimentDesign:
    def __init__(self, options=None, conf_path=None):
        self.config = options or Option(conf_path)
        self.train_loader = None
        self.test_loader = None
        self.model = None
        self.trainer = None
        self.start_epoch = 0
        self.test_input = None

        os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID" 
        os.environ['CUDA_VISIBLE_DEVICES'] = self.config.visible_devices 

        self.config.set_save_path()
        self.logger = self.set_logger()
        self.config.print_parameter()

        self.prepare()

    def set_logger(self):
        logger = Logger(**self.config)
        return logger

    def prepare(self):
        self._set_gpu()
        self._set_dataloader()
        self._set_model()
        self._replace()
        self.logger.info(self.model)
        self._set_trainer()

    def _set_gpu(self):
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed(self.config.seed)
        assert self.config.GPU <= torch.cuda.device_count() - 1, "Invalid GPU ID"
        cudnn.benchmark = True
        # 추 후 DDP 추가

    def _set_dataloader(self):
    # create data loader
        self.train_loader, self.test_loader, self.n_classes, self.img_size = create_loader(self.config.batch_size, self.config.data_dir, self.config.dataset)

    def _set_model(self):

        if self.config.dataset in ["imagenet"]:
            self.test_input = Variable(torch.randn(1, 3, 224, 224).cuda())
            self.model = timm.create_model(self.config.network, pretrained=True)
        else:
            assert False, "unsupport data set: " + self.config.dataset

    def _set_optimizer(self):
        self.optim, self.scheduler, self.criterion = get_optimizer_scheduler(self.model, self.config)
        
        

    def _set_trainer(self):
    # set lr master
        # set trainer
        if self.config.trainer =="default":
            self.trainer = Trainer(
                model=self.model,
                model_teacher=self.model_teacher,
                generator = self.generator,
                train_loader=self.train_loader,
                test_loader=self.test_loader,
                lr_master_S=lr_master_S,
                lr_master_G=lr_master_G,
                settings=self.config,
                logger=self.logger,
                opt_type=self.config.opt_type,
                optimizer_state=self.optimizer_state,
                run_count=self.start_epoch)