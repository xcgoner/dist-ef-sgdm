#coding=utf-8
import torch
from torch.optim import Optimizer
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors, \
    _take_tensors
import time
import os
import random
# ER SGD with blockwise sparsification


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

        self.all_reduce = all_reduce
        self.nesterov = args.nesterov
        self.log_writer = log_writer

        # error reset
        self.reset_interval = args.reset_interval
        self.reset_counter = 0
        self.sparse_ratio = args.sparse_ratio
        self.nodes = dist.get_world_size()
        random.seed(args.seed)

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

        self.MB = 1024 * 1024
        self.bucket_size = 50 * self.MB

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

        block_counter = 0
        sparse_counter = 0

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']

            cur_lr = group['lr']

            all_grads = []

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad.data
                if self.compression_buffer==False:
                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)

                all_grads.append(d_p)

            length = 0
            for _ in _take_tensors(all_grads, self.bucket_size):
                length += 1

            dev_grads_buckets = _take_tensors(all_grads, self.bucket_size)
            for i, dev_grads in enumerate(dev_grads_buckets):

                block_counter += 1

                if random.uniform(0, 1) < self.sparse_ratio:
                    sparse_counter += 1
                    continue
                
                d_p_new = _flatten_dense_tensors(dev_grads)

                dist.all_reduce(d_p_new, op=dist.ReduceOp.SUM)
                d_p_new /= self.nodes

                #unflatten
                dev_grads_new = _unflatten_dense_tensors(d_p_new,dev_grads)
                for grad, reduced in zip(dev_grads, dev_grads_new):
                    grad.copy_(reduced)

            for p in group['params']:
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
        
        print("Sparsification: {}/{}".format(sparse_counter, block_counter))

        return loss

