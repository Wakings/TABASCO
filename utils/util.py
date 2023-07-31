#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from itertools import count
import os
from numpy import random
import torch
import platform
import numpy as np
from torch.autograd import Variable
import copy

systype = platform.system()


class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def set_tensor(tensor_var, boolen, device):
	# print(tensor_var)
	tensor_var = tensor_var.to(device)
	# tensor_var = tensor_var.to(device,non_blocking=True)  
	#return Variable(tensor_var, requires_grad=boolen)
	tensor_var.requires_grad = boolen
	return tensor_var

