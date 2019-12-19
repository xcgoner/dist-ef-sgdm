#coding=utf-8
import torch
from torch.optim import Optimizer
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors, \
    _take_tensors
import time
import os
#Signum with majority vote


class SGD_distribute(Optimizer):

    def __init__(self, params, args, log_writer, **kwargs):

        lr = args.lr
        momentum = args.momentum
        weight_decay = args.weight_decay
        compression_buffer = args.compress
        all_reduce = args.all_reduce
        local_rank = args.local_rank
        gpus_per_machine = args.gpus_per_machine

        self.err_buf = []
        self.prev_lr = 0
        self.server_err_buf = []

        self.nesterov = args.nesterov
        self.log_writer = log_writer

        # error reset
        self.reset_interval = args.reset_interval
        self.reset_counter = 0

        self.args = args

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum,
                        weight_decay=weight_decay)

        super(SGD_distribute, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD_distribute, self).__setstate__(state)

    
    def average_params(self):
        # synchronize model parameters
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                dist.all_reduce(p.data, op=dist.ReduceOp.SUM)
                p.data /= self.nodes
    
    def average_momentum(self):
        # synchronize momentum
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                param_state = self.state[p]
                buf = param_state['momentum_buffer']
                dist.all_reduce(buf, op=dist.ReduceOp.SUM)
                buf /= self.nodes


    def step(self, closure=None):

        # error reset
        self.reset_counter += 1

        args = self.args

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']

            cur_lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                if momentum != 0:
                    # momentum
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                    else:
                        buf = param_state['momentum_buffer']

                    buf.mul_(momentum).add_(p.grad.data)
                    if weight_decay != 0:
                        buf.add_(weight_decay, p.data)
                    if self.nesterov:
                        p.grad.data.add_(momentum, buf)
                    else:
                        p.grad.data.copy_(buf)

                p.data.add_(-group['lr'], p.grad.data)

            # error reset
            if self.reset_counter == self.reset_interval:
                self.average_params()
                self.average_momentum()
                self.reset_counter = 0

        return loss

