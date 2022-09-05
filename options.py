
import os
import shutil
from pathlib import Path

from pyhocon import ConfigFactory

class DefaultOption(object):
    def __init__(self):
        # general option
        self.save_path = "./save/"
        self.data_path = "/dataset/"
        self.dataset = "cifar10"
        self.seed = 0
        self.nGPU = 1
        self.gpu = 0
        
        # dataloader option
        self.worker = 4

        # training option
        self.train = False

        # optimization option
        self.epcohs = 200
        self.batch_size = 128
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.optimizer = "SGD"

        self.LR = 0.01
        self.lr_policy = "multi_step"
        self.power = 1
        self.step = [0.6, 0.8]
        self.endlr = 0.0001
        self.lr_gamma = 0.1

        # model option


class Option(DefaultOption):
    def __init__(self, conf_path):
        super(Option, self).__init__()
        self.conf = ConfigFactory.parse_file(conf_path)
        self.save_path = self.conf['save_path']
        self.seed = getattr(self.conf, "seed", 0)
        self.nGPU = getattr(self.conf, "nGPU", 0)
        self.workers = getattr(self.conf, "workers", 4)
    
        # distributed setting
        self.dist = getattr(self.conf, "dist", False)
    
        # Train dataset options
        self.dataset = self.conf['dataset']
        self.data_path = self.conf['data_path'] # dataset path
        self.epochs = self.conf['epochs']
        self.batch_size = self.conf['batch_size']
        self.test_batch_size= getattr(self.conf, 'test_batch_size', self.batch_size)
        self.pin_memory = getattr(self.conf, 'pin_memory', False)
        
        # Train model option
        self.arch = self.conf['arch']
        self.criterion = getattr(self.conf, 'criterion', "CE")
        
        # model optimizer option
        self.weight_decay = self.conf['weight_decay']
        self.optim = self.conf['optim']
        self.grad_clip = getattr(self.conf, 'grad_clip', 5)
        self.unrolled = getattr(self.conf, 'unrolled', False)

        if self.optim.lower() == "sgd":
            self.momentum = getattr(self.conf, 'momentum', 0.9)
            self.nesterov = getattr(self.conf, 'netserov', False)
        
        elif self.optim.lower() == "adam":
            self.betas = getattr(self.conf, 'betas', [0.9, 0.999])

        self.warmup = getattr(self.conf, "warmup", 5)
        self.LR = getattr(self.conf, "lr", 0.01)
        self.scheduler = getattr(self.conf, "scheduler", "multistep") # default = multistep
        
        if self.scheduler == 'multistep':
            self.milestones = getattr(self.conf, "ml_step", [60, 90])
            self.gammas = getattr(self.conf, "lr_step", 0.1)
            if type(self.gammas) == float:
                self.gammas = [self.gammas for _ in range(len(self.milestones))]
                
        else:
            self.gammas = getattr(self.conf, "lr_gamma", 0.1)
            self.eta_min = getattr(self.conf, "eta_min", 0.001)
 
        self.log_override = False       
        # logger option
        self.tag = getattr(self.conf, "tag", "default")
        self.print_freq= self.conf.print_freq
        del self.conf
    
    def set_save_path(self):
        self.save_path = os.path.join(self.save_path)
        if os.path.exists(self.save_path) :
            print(f"{self.save_path} is exists")
            if self.log_override:
                shutil.rmtree(self.save_path)
            else:
                print(f"load log path {self.save_path}")
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def print_parameters(self):
        for key, value in sorted(self.__dict__.items()):
            print(f"{key} : {value}")
