from pathlib import Path
import importlib, warnings
import os, sys, time, numpy as np
from torch.utils.tensorboard import SummaryWriter

class Logger(object):
    def __init__(self, save_path, seed, create_model_dir=True, use_tfb=False, use_wandb=False, **kwargs):
        """Create a summary writer logging to log_dir."""
        self.seed = int(seed)
        self.log_dir = Path(save_path) / "log"
        self.model_dir = Path(save_path) / "checkpoint"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if create_model_dir:
            self.model_dir.mkdir(parents=True, exist_ok=True)
        # self.meta_dir.mkdir(mode=0o775, parents=True, exist_ok=True)
        
        self.use_tfb = use_tfb
        self.use_wandb = use_wandb
        self.logger_path = self.log_dir / "seed-{:}-T-{:}.log".format(
            self.seed, time.strftime("%d-%h-at-%H-%M-%S", time.gmtime(time.time()))
        )
        self.logger_file = open(self.logger_path, "w")
        
        if self.use_tfb:
            self.tensorboard_dir = self.log_dir / (
                "tensorboard-{:}".format(time.strftime("%d-%h", time.gmtime(time.time())))
            )
            # self.tensorboard_dir = self.log_dir / ('tensorboard-{:}'.format(time.strftime( '%d-%h-at-%H:%M:%S', time.gmtime(time.time()) )))

            self.tensorboard_dir.mkdir(mode=0o775, parents=True, exist_ok=True)
            self.writer = SummaryWriter(str(self.tensorboard_dir))
        else:
            self.writer = None

    def __repr__(self):
        return "{name}(dir={log_dir}, use-tfb={use_tfb}, use-wandb={use_wandb},writer={writer})".format(
            name=self.__class__.__name__, **self.__dict__
        )

    def path(self, mode):
        valids = ("model", "best", "info", "log", None)
        if mode is None:
            return self.log_dir
        elif mode == "model":
            return self.model_dir / "seed-{:}-basic.pth".format(self.seed)
        elif mode == "best":
            return self.model_dir / "seed-{:}-best.pth".format(self.seed)
        elif mode == "info":
            return self.log_dir / "seed-{:}-last-info.pth".format(self.seed)
        elif mode == "log":
            return self.log_dir
        else:
            raise TypeError("Unknow mode = {:}, valid modes = {:}".format(mode, valids))

    def extract_log(self):
        return self.logger_file

    def close(self):
        self.logger_file.close()
        if self.writer is not None:
            self.writer.close()

    def log(self, string, save=True, stdout=False):
        if stdout:
            sys.stdout.write(string)
            sys.stdout.flush()
        else:
            print(string)
        if save:
            self.logger_file.write("{:}\n".format(string))
            self.logger_file.flush()

    def scalar_summary(self, scalar_dict, step):
        """Log a scalar variable."""
        if not self.use_tfb:
            warnings.warn("Do set use-tensorflow installed but call scalar_summary")
        else:
            assert isinstance(scalar_dict, dict), "please input dictionary class"
            
            for tag, value in scalar_dict.items():
                self.writer.add_scalar(tag, value, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""
        import scipy

        if not self.use_tfb:
            warnings.warn("Do set use-tensorflow installed but call scalar_summary")
            return

        img_summaries = []
        print("future work import image library")

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""
        print("future work import image library")