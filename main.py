import argparse
import logging, os, shutil
import time
import traceback
import sys
import copy
import io
from datetime import timedelta
from contextlib import contextmanager
import numpy as np
import random
# get torch library

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

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

        self.config.set_save_path()
        self.logger = self.set_logger()
        self.config.print_parameters()

        self.prepare()
    
    def set_logger(self):
        logger = Logger(**self.config.__dict__)
        return logger

    def prepare(self):
        #self._init_distributed()
        self._set_gpu()
        self._set_dataloader()
        self._set_model()
        self._set_optimizer()
        self._set_checkpoint()
        self.logger.log(self.model)
        self._set_trainer()

    def _init_distributed(self):
        self.dist_url = "env://"
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.device_id = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(
            backend='nccl',
            init_method=self.dist_url,
            timeout=timedelta(seconds=10)
        )
        print(f"=> set cuda device = {self.device_id}")

        torch.cuda.set_device(self.device_id)
        dist.barrier()

    def _set_gpu(self):
        
        if self.config.dist:
            torch.manual_seed(self.config.seed)
            torch.cuda.manual_seed(self.config.seed)
            random.seed(self.config.seed)
            self._init_distributed()
            cudnn.benchmark = True
            # 추 후 DDP 추가
        else:
            torch.manual_seed(self.config.seed)
            torch.cuda.manual_seed(self.config.seed)
            assert self.config.GPU <= torch.cuda.device_count() - 1, "Invalid GPU ID"
            cudnn.benchmark = True
            self.device_id = 0
            
    def _set_dataloader(self):
    # create data loader
        self.train_loader, self.test_loader, self.n_classes, self.img_size = create_loader(self.config)
    def _set_model(self):

        if self.config.dataset.lower() in ["imagenet"]:
            self.test_input = Variable(torch.randn(1, 3, 224, 224).cuda())
            self.model = timm.create_model(self.config.arch, pretrained=True)
            if self.config.dist:
                self.model.cuda(self.device_id)
                self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
                self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.device_id])

        else:
            assert False, "unsupport data set: " + self.config.dataset

    def _set_optimizer(self):
        self.optim, self.scheduler, self.criterion = get_optimizer_scheduler(self.model, self.config)
    
    def _set_checkpoint(self):
        self.state = State(self.config.arch, self.model, self.optim, self.scheduler)
        self.state = load_checkpoint(self.config.checkpoint_path, self.device_id, self.state)
        
        
    def _set_trainer(self):
    # set lr master
    # set trainer
        self.trainer = Trainer(
            model=self.model,
            train_loader=self.train_loader,
            test_loader=self.test_loader,
            loss = self.criterion,
            optimizer = self.optim,
            options=self.config,
            logger=self.logger,
            run_count=self.start_epoch)
    
    def train(self):
        start_epoch = self.state.epoch + 1
        print("start Experiment train")
        for epoch in range(start_epoch, self.config.epochs):
            train_loss, train_top1, train_top5 =self.trainer.train(epoch, self.device_id)
            test_loss, test_top1, test_top5 = self.trainer.test(epoch)
                     
            if self.device_id == 0:
                self.state.epoch = epoch
                is_best = test_top1 > self.state.best_acc1
                self.state.best_acc1 = max(test_top1, self.state.best_acc1)
                save_checkpoint(self.state, is_best, self.config.checkpoint_path)
                

parser = argparse.ArgumentParser(description="PyTorch Elastic ImageNet Training")
parser.add_argument("--conf_path", type=str, default="default.hocon", help="set conf path")
parser.add_argument("--resume", type=bool, default=False, help="load checkpoint")
parser.add_argument("--checkpoint_path", default=None, help="checkpoint path")


def main():
    
    args = parser.parse_args()
    option = Option(args.conf_path)
    option.set_save_path()
    if args.resume:
        if args.checkpoint_path is not None:
            option.checkpoint_path = args.checkpoint_path
        else:
            option.checkpoint_path = option.save_path + "/last_checkpoint.pth"
    else:
        option.checkpoint_path = option.save_path + "/last_checkpoint.pth"
    
    experiment = ExperimentDesign(option)
    experiment.train()
    
class State:
    """
    Container for objects that we want to checkpoint. Represents the
    current "state" of the worker. This object is mutable.
    """

    def __init__(self, arch, model, optimizer, scheduler):
        self.epoch = -1
        self.best_acc1 = 0
        self.arch = arch
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def capture_snapshot(self):
        """
        Essentially a ``serialize()`` function, returns the state as an
        object compatible with ``torch.save()``. The following should work
        ::
        snapshot = state_0.capture_snapshot()
        state_1.apply_snapshot(snapshot)
        assert state_0 == state_1
        """
        return {
            "epoch": self.epoch,
            "best_acc1": self.best_acc1,
            "arch": self.arch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict()
        }

    def apply_snapshot(self, obj, device_id):
        """
        The complimentary function of ``capture_snapshot()``. Applies the
        snapshot object that was returned by ``capture_snapshot()``.
        This function mutates this state object.
        """

        self.epoch = obj["epoch"]
        self.best_acc1 = obj["best_acc1"]
        self.state_dict = obj["state_dict"]
        self.model.load_state_dict(obj["state_dict"])
        self.optimizer.load_state_dict(obj["optimizer"])
        self.scheduler.load_state_dict(obj['scheduler'])

    def save(self, f):
        torch.save(self.capture_snapshot(), f)

    def load(self, f, device_id):
        # Map model to be loaded to specified single gpu.
        snapshot = torch.load(f, map_location=f"cuda:{device_id}")
        self.apply_snapshot(snapshot, device_id)


def load_checkpoint(
    checkpoint_file: str,
    device_id: int,
    state: State # SGD
) -> State:
    """
    Loads a local checkpoint (if any). Otherwise, checks to see if any of
    the neighbors have a non-zero state. If so, restore the state
    from the rank that has the most up-to-date checkpoint.
    .. note:: when your job has access to a globally visible persistent storage
              (e.g. nfs mount, S3) you can simply have all workers load
              from the most recent checkpoint from such storage. Since this
              example is expected to run on vanilla hosts (with no shared
              storage) the checkpoints are written to local disk, hence
              we have the extra logic to broadcast the checkpoint from a
              surviving node.
    """

    if os.path.isfile(checkpoint_file):
        print(f"=> loading checkpoint file: {checkpoint_file}")
        state.load(checkpoint_file, device_id)
        print(f"=> loaded checkpoint file: {checkpoint_file}")

    # logic below is unnecessary when the checkpoint is visible on all nodes!
    # create a temporary cpu pg to broadcast most up-to-date checkpoint
    with tmp_process_group(backend="gloo") as pg:
        rank = dist.get_rank(group=pg)

        # get rank that has the largest state.epoch
        epochs = torch.zeros(dist.get_world_size(), dtype=torch.int32)
        epochs[rank] = state.epoch
        dist.all_reduce(epochs, op=dist.ReduceOp.SUM, group=pg)
        t_max_epoch, t_max_rank = torch.max(epochs, dim=0)
        max_epoch = t_max_epoch.item()
        max_rank = t_max_rank.item()

        # max_epoch == -1 means no one has checkpointed return base state
        if max_epoch == -1:
            print(f"=> no workers have checkpoints, starting from epoch 0")
            return state

        # broadcast the state from max_rank (which has the most up-to-date state)
        # pickle the snapshot, convert it into a byte-blob tensor
        # then broadcast it, unpickle it and apply the snapshot
        print(f"=> using checkpoint from rank: {max_rank}, max_epoch: {max_epoch}")

        with io.BytesIO() as f:
            torch.save(state.capture_snapshot(), f)
            raw_blob = np.frombuffer(f.getvalue(), dtype=np.uint8)

        blob_len = torch.tensor(len(raw_blob))
        dist.broadcast(blob_len, src=max_rank, group=pg)
        print(f"=> checkpoint broadcast size is: {blob_len}")

        if rank != max_rank:
            # pyre-fixme[6]: For 1st param expected `Union[List[int], Size,
            #  typing.Tuple[int, ...]]` but got `Union[bool, float, int]`.
            blob = torch.zeros(blob_len.item(), dtype=torch.uint8)
        else:
            blob = torch.as_tensor(raw_blob, dtype=torch.uint8)

        dist.broadcast(blob, src=max_rank, group=pg)
        print(f"=> done broadcasting checkpoint")

        if rank != max_rank:
            with io.BytesIO(blob.numpy()) as f:
                snapshot = torch.load(f)
            state.apply_snapshot(snapshot, device_id)

        # wait till everyone has loaded the checkpoint
        dist.barrier(group=pg)

    print(f"=> done restoring from previous checkpoint")
    return state

@contextmanager
def tmp_process_group(backend):
    cpu_pg = dist.new_group(backend=backend)
    try:
        yield cpu_pg
    finally:
        dist.destroy_process_group(cpu_pg)
        
def save_checkpoint(state: State, is_best: bool, filename: str):
    checkpoint_dir = os.path.dirname(filename)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # save to tmp, then commit by moving the file in case the job
    # gets interrupted while writing the checkpoint
    tmp_filename = filename + ".tmp"
    torch.save(state.capture_snapshot(), tmp_filename)
    os.rename(tmp_filename, filename)
    print(f"=> saved checkpoint for epoch {state.epoch} at {filename}")
    if is_best:
        best = os.path.join(checkpoint_dir, "model_best.pth.tar")
        print(f"=> best model found at epoch {state.epoch} saving to {best}")
        shutil.copyfile(filename, best)
        

if __name__ == "__main__":
    main()