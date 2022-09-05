"""
basic trainer
"""
import time

import torch.autograd
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import utils as utils
from utils import Logger, AverageMeter, compute_accuracy
import numpy as np
import torch
import math
import os, copy

"""
추 후 추가해야 할 모듈
DP, DDP를 고려한 trainer 생성 필요
공부해야 할 수 있을 듯?
"""

__all__ = ["Trainer"]

class Trainer(object):
	"""
	trainer for training network, use SGD
	"""

	def __init__(self, model, train_loader, test_loader, loss, optimizer, options, logger, run_count=0):
		"""
		init trainer
		"""
		
		self.config = options
		self.model = model
		self.train_loader = train_loader
		self.test_loader = test_loader
		self.loss = loss
		self.optimizer = optimizer
		self.logger = logger
		self.run_count = run_count
		self.scalar_info = {}
		self.total_batch_iter = len((self.train_loader))


	def forward(self, images, labels):
		"""
		forward propagation
		"""
		# forward and backward and optimize

		output = self.model(images)
		if labels is not None:
			loss = self.loss(output, labels)
			return output, loss
		else:
			return output, None

	def backward(self, loss):
		"""
		backward propagation
		"""
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

	def train(self, epoch, device_id):
		"""
		training
		"""
		total_top1_acc = AverageMeter()
		total_loss = AverageMeter()
		total_top5_acc = AverageMeter()


		self.model.train()
		if self.config.dist:
			self.train_loader.batch_sampler.sampler.set_epoch(epoch)
		start_time = time.time()
		end_time = start_time
  
		for idx, (images, labels) in enumerate(self.train_loader):
			images = images.cuda(device_id, non_blocking=True)
			labels = labels.cuda(device_id, non_blocking=True)
			output, loss = self.forward(images, labels)
			self.backward(loss)
			# compute accuracy
			top1_acc, top5_acc = compute_accuracy(output, labels, topk=[1, 5])
			total_top1_acc.update(top1_acc.item(), images.size(0))
			total_top5_acc.update(top5_acc.item(), images.size(0))
			total_loss.update(loss.item(), images.size(0))			
			if idx % self.config.print_freq == 0 and device_id == 0:
				batch_result = "[Epoch {}/{}] [Batch {}/{}] [loss : {:.4f}] [top1 acc : {:.4f}] [top5 acc : {:.4f}]".format(
					epoch, self.config.epochs, idx, self.total_batch_iter, total_loss.avg, total_top1_acc.avg, total_top5_acc.avg
				)
				self.logger.log(batch_result)
		end_time = time.time()

		epoch_time = (end_time - start_time)

		total_result = "[Epoch {}/{}] End \n [epoch time : {:.4f}] [loss : {:.4f}] [top1 acc : {:.4f}] [top5 acc : {:.4f}]".format(
					epoch, self.config.epochs, epoch_time, total_loss.avg, total_top1_acc.avg, total_top5_acc.avg
				)
		
		self.logger.log(total_result)
  
		# 추 후 tensorboard update


		return total_loss.avg, total_top1_acc.avg, total_top5_acc.avg
		

	def test(self, epoch):
		"""
		testing
		"""
		total_top1_acc = AverageMeter()
		total_loss = AverageMeter()
		total_top5_acc = AverageMeter()
		
		self.model.eval()
		
		total_batch_iter = len(self.test_loader)
		start_time = time.time()
		start_str = "[Epoch {}/{}] validation start".format(
					epoch, self.config.epochs)
		
		self.logger.log(start_str)

		with torch.no_grad():
			for idx, (images, labels) in enumerate(self.test_loader):
				start_time = time.time()
				
				labels = labels.cuda()
				images = images.cuda()
				output, loss = self.forward(images, labels)
			top1_acc, top5_acc = compute_accuracy(output, labels, topk=[1, 5])
			total_top1_acc.update(top1_acc.item(), images.size(0))
			total_top5_acc.update(top5_acc.item(), images.size(0))
			total_loss.update(loss.item(), images.size(0))			

			if idx % self.config.print_freq == 0:
				batch_result = "[Epoch {}/{}] [Batch {}/{}] [loss : {:.4f}] [top1 acc : {:.4f}] [top5 acc : {:.4f}]".format(
					epoch, self.config.epochs, idx, total_batch_iter, total_loss.avg, total_top1_acc.avg, total_top5_acc.avg
				)
				self.logger.log(batch_result)
		end_time = time.time()

		epoch_time = (end_time - start_time)

		total_result = "[Epoch {}/{}] End \n [epoch time : {:.4f}] [loss : {:.4f}] [top1 acc : {:.4f}] [top5 acc : {:.4f}]".format(
					epoch, self.config.epochs, epoch_time, total_loss.avg, total_top1_acc.avg, total_top5_acc.avg
				)
		
		self.logger.log(total_result)
  
		# 추 후 tensorboard update


		return total_loss.avg, total_top1_acc.avg, total_top5_acc.avg
