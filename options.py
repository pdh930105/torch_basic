
import os
import shutil

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
        self.data_path = self.conf['data_path']
        self.dataset = self.conf['dataset']
        self.seed = getattr(self.conf, "seed", 0)
        self.nGPU = getattr(self.conf, "nGPU", 0)
        self.gpu = getattr(self.conf, "GPU", 0)
        self.workers = getattr(self.conf, "workers", 4)
        self.name = self.conf['name']
        # distributed setting
        self.distributed = getattr(self.conf, "distributed", False)
        
        # NAS Option
        self.search_space = self.conf['search_space']
        self.max_nodes = self.conf['max_nodes']
        self.channel = self.conf['channel']
        self.num_stages = getattr(self.conf, 'num_stages', None)
        self.stage_types = getattr(self.conf, 'stage_types', None)
        self.track_running_stats = self.conf['track_running_stats']
        self.affine = self.conf['affine']
        self.portion = getattr(self.conf, 'portion', 0.5)
        self.criterion = getattr(self.conf, 'criterion', "Softmax")
        
        self.set_mode_epoch = getattr(self.conf, 'set_mode_epoch', [15, 30, 45, 60])

        # Train options
        self.rand_seed = self.conf['rand_seed']
        self.train = self.conf['train']
        self.epochs = self.conf['epochs']
        self.batch_size = self.conf['batch_size']
        self.test_batch_size= self.conf['test_batch_size']
        
        # model optimizer option
        self.decay = self.conf['weight_decay']
        self.optim = self.conf['optim']
        self.grad_clip = getattr(self.conf, 'grad_clip', 5)
        self.unrolled = getattr(self.conf, 'unrolled', False)
        
        # arch optimizer option
        self.arch_learning_rate = getattr(self.conf, 'arch_learning_rate', 1e-3)
        self.arch_weight_decay = getattr(self.conf, 'arch_weight_decay', 0)

        if self.optim.lower() == "sgd":
            self.momentum = self.conf['momentum']
            self.nesterov = self.conf['nesterov']
        
        elif self.optim.lower() == "adam":
            self.adam_alpha = self.conf['adam_alpha']
            self.adam_beta = self.conf['adam_beta']

        self.warmup = getattr(self.conf, "warmup", 5)
        
        self.LR = getattr(self.conf, "lr", 0.01)
        self.scheduler = getattr(self.conf, "scheduler", "multistep") # default = multistep
        if self.scheduler == 'multi_step':
            self.milestones = getattr(self.conf, "ml_step", [60, 90])
        self.gammas = getattr(self.conf, "lr_gamma", 0.1)
        self.eta_min = getattr(self.conf, "eta_min", 0.001)
        self.log_override = False
        
        # switchable option
        self.switchable = getattr(self.conf, 'switchable', False)
        if self.switchable:
            self.bits_list = sorted(self.conf['bits_list'])
            
        
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
