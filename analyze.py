import torch
import numpy as np
import torch.nn as nn
from numpy.linalg import eigvals
from torch.autograd import Variable
from copy import deepcopy
import logging


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Analyzer(object):
    def __init__(self, model, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.weight_decay = args.arch_weight_decay
        self.hessian = None
        self.grads = None
        self.adaptive_stop = args.adaptive_stop
        self.adas = args.adas

    def _compute_unrolled_model(self, input, target, lr_vector, layers_todo, network_optimizer):
        loss = self.model._loss(input, target)
        theta = _concat(self.model.parameters()).data
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(
                self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        # dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay * theta
        model_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        dtheta = _concat([grad_i + self.network_weight_decay * theta_i
                          for grad_i, theta_i in
                          zip(torch.autograd.grad(loss, model_params), model_params)])

        ################################################################################
        # AdaS
        # adaptive stopping: frozen parameters don't have gradients,
        # so don't update them
        if self.adas:
            iteration_p = 0
            offset_p = 0
            offset_dp = 0
            for p in self.model.parameters():
                p_length = np.prod(p.size())

                if ~layers_todo[iteration_p]:
                    # not updating the frozen conv layers
                    iteration_p += 1
                    offset_p += p_length
                    continue
                lr = lr_vector[iteration_p]
                d_p = moment[offset_p: offset_p + p_length] + \
                      dtheta[offset_dp: offset_dp + p_length]
                theta[offset_p: offset_p + p_length].sub_(d_p, alpha=lr)
                offset_p += p_length
                offset_dp += p_length
                iteration_p += 1
            unrolled_model = self._construct_model_from_theta(theta, layers_todo)
        ################################################################################
        # original darts
        else:
            unrolled_model = self._construct_model_from_theta(theta.sub(lr_vector, moment + dtheta), None)

        return unrolled_model

    def _backward_step(self, input_valid, target_valid, create_graph):
        loss = self.model._loss(input_valid, target_valid)
        loss.backward(create_graph=create_graph)

    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid,
                                lr, layers, network_optimizer, create_graph):
        # eqn(6)：dαLval(w',α) ，where w' = w − ξ*dwLtrain(w, α)
        # w'
        unrolled_model = self._compute_unrolled_model(input_train, target_train, lr, layers,
                                                      network_optimizer)  # unrolled_model: w -> w'
        # Lval: validation loss
        unrolled_loss = unrolled_model._loss(input_valid, target_valid)

        unrolled_loss.backward(create_graph=create_graph)
        # dαLval(w',α)
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]  # grad wrt alpha

        # dw'Lval(w',α)
        vector = [v.grad.data if v.requires_grad else None for v in
                  unrolled_model.parameters()]  # unrolled_model.parameters(): w‘
        ################################################################################
        # AdaS: use different etas for different w's
        # with adaptive stopping
        if self.adas:
            iteration_p = 0
            for p in vector:
                if ~layers[iteration_p]:
                    assert p is None
                    iteration_p += 1
                    continue
                p.mul_(lr[iteration_p])
                iteration_p += 1
        ################################################################################

        # eqn(8): (dαLtrain(w+,α)-dαLtrain(w-,α))/(2*epsilon)
        # where w+=w+dw'Lval(w',α)*epsilon w- = w-dw'Lval(w',α)*epsilon
        implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

        # eqn(6)-eqn(8): dαLval(w',α)-(dαLtrain(w+,α)-dαLtrain(w-,α))/(2*epsilon)
        for g, ig in zip(dalpha, implicit_grads):
            # g.data.sub_(eta, ig.data)
            g.data.sub_(ig.data)
        # update α
        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    def _construct_model_from_theta(self, theta, layers_todo):
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)

        ################################################################################
        # adaptive stopping
        if self.adaptive_stop:
            iteration_p = 0
            for p in model_new.parameters():
                if ~layers_todo[iteration_p]:
                    p.requires_grad = False
                    p.grad = None
                iteration_p += 1
        ################################################################################

        return model_new.cuda()

    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)
        loss = self.model._loss(input, target)
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(2 * R, v)
        loss = self.model._loss(input, target)
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]

    def compute_dw(self, input_train, target_train, input_valid, target_valid,
                   lr, layers, network_optimizer, unrolled):
        self.zero_grads(self.model.parameters())
        self.zero_grads(self.model.arch_parameters())
        if unrolled:
            self._backward_step_unrolled(input_train, target_train, input_valid, target_valid,
                                         lr, layers, network_optimizer, False)
        else:
            self._backward_step(input_valid, target_valid, False)

        self.grads = [v.grad + self.weight_decay * v for v in self.model.arch_parameters()]
        return self.grads

    def compute_Hw(self, input_train, target_train, input_valid, target_valid,
                   lr, layers, network_optimizer, unrolled):
        # logging.info('zero grad model param')
        self.zero_grads(self.model.parameters())
        # logging.info('zero grad arch param')
        self.zero_grads(self.model.arch_parameters())
        # if unrolled:
        #    self._backward_step_unrolled(input_train, target_train, input_valid, target_valid,
        #                                 lr, layers, network_optimizer, True)
        # else:
        #    self._backward_step(input_valid, target_valid, True)

        # self.grads = [v.grad + self.weight_decay*v for v in self.model.arch_parameters()]
        # logging.info('compute loss')
        loss = self.model._loss(input_valid, target_valid)
        # logging.info('compute hessian')
        self.hessian = self._hessian(loss, self.model.arch_parameters())
        return self.hessian

    def compute_eigenvalues(self):
        # hessian = self.compute_Hw(input, target)
        if self.hessian is None:
            raise ValueError
        return eigvals(self.hessian.cpu().data.numpy())

    def zero_grads(self, parameters):
        for p in parameters:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
                # if p.grad.volatile:
                #    p.grad.data.zero_()
                # else:
                #    data = p.grad.data
                #    p.grad = Variable(data.new().resize_as_(data).zero_())

    def gradient(self, _outputs, _inputs, grad_outputs=None, retain_graph=None,
                 create_graph=False):
        if torch.is_tensor(_inputs):
            _inputs = [_inputs]
        else:
            _inputs = list(_inputs)
        grads = torch.autograd.grad(_outputs, _inputs, grad_outputs,
                                    allow_unused=True,
                                    retain_graph=retain_graph,
                                    create_graph=create_graph)
        grads = [x if x is not None else torch.zeros_like(y) for x, y in zip(grads,
                                                                             _inputs)]
        return torch.cat([x.contiguous().view(-1) for x in grads])

    def _hessian(self, outputs, inputs, out=None, allow_unused=False,
                 create_graph=False):
        # assert outputs.data.ndimension() == 1

        if torch.is_tensor(inputs):
            inputs = [inputs]
        else:
            inputs = list(inputs)

        n = sum(p.numel() for p in inputs)
        if out is None:
            out = Variable(torch.zeros(n, n)).type_as(outputs)

        ai = 0
        for i, inp in enumerate(inputs):
            # logging.info('input {}'.format(i))
            # logging.info('grad')
            [grad] = torch.autograd.grad(outputs, inp, create_graph=True,
                                         allow_unused=allow_unused)
            grad = grad.contiguous().view(-1) + self.weight_decay * inp.view(-1)
            # grad = outputs[i].contiguous().view(-1)

            for j in range(inp.numel()):
                # logging.info('input {}\'s{}'.format(i,j))
                # print('(i, j): ', i, j)
                if grad[j].requires_grad:
                    # logging.info('grad grad')
                    row = self.gradient(grad[j], inputs[i:], retain_graph=True)[j:]
                else:
                    n = sum(x.numel() for x in inputs[i:]) - j
                    row = Variable(torch.zeros(n)).type_as(grad[j])
                    # row = grad[j].new_zeros(sum(x.numel() for x in inputs[i:]) - j)

                out.data[ai, ai:].add_(row.clone().type_as(out).data)  # ai's row
                if ai + 1 < n:
                    out.data[ai + 1:, ai].add_(row.clone().type_as(out).data[1:])  # ai's column
                del row
                ai += 1
            del grad
        return out
