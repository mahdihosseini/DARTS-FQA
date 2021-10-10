import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):
    def __init__(self, model, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.adas = args.adas
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                          lr=args.arch_learning_rate, betas=(0.5, 0.999),
                                          weight_decay=args.arch_weight_decay)
        self.gumbel = args.gumbel
        self.adaptive_stop = args.adaptive_stop
        self.grad_clip = args.grad_clip
        # Necessary for PLCC Criterion
        self.batch_size = args.batch_size

    def _compute_unrolled_model(self, input, target, lr_vector, layers_todo, network_optimizer):
        loss = self.model._loss(input, target, self.gumbel)
        theta = _concat(self.model.parameters()).data
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(
                self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        # dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay * theta

        # adaptive stopping:
        # filter out params that are frozen (requires_grad = False)
        model_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        
        ################################################################################
        # using gumbel-softmax:
        # for unused ops there will be no gradient and this needs to be handled
        if self.gumbel:
            dtheta = _concat([grad_i + self.network_weight_decay * theta_i if grad_i is not None
                              else self.network_weight_decay * theta_i
                              for grad_i, theta_i in
                              zip(torch.autograd.grad(loss, model_params, allow_unused=True), model_params)])
        ##########################
        # not using gumbel-softmax
        else:
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
        ################
        # original darts
        else:
            unrolled_model = self._construct_model_from_theta(theta.sub(lr_vector, moment + dtheta), None)
        ################################################################################
        
        return unrolled_model

    def step(self, input_train, target_train, input_valid, target_valid, lr, layers, network_optimizer, unrolled):
        self.optimizer.zero_grad()
        if unrolled:
            self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, lr, layers, network_optimizer)
        else:
            # Adjustment for PLCC Criterion
            self._backward_step(input_valid, target_valid)

            # Previous
            # self._backward_step(input_valid, target_valid)

        # Add gradient clipping because gumbel softmax leads to gradients with high magnitude
        if self.gumbel:
            torch.nn.utils.clip_grad_norm_(self.model.arch_parameters(), self.grad_clip)
            
        self.optimizer.step()

    def _backward_step(self, input_valid, target_valid):
        loss = self.model._loss(input_valid, target_valid, self.gumbel)
        loss.backward()

    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, lr, layers, network_optimizer):
        # eqn(6)：dαLval(w',α) ，where w' = w − ξ*dwLtrain(w, α)
        # w'
        unrolled_model = self._compute_unrolled_model(input_train, target_train, lr, layers,
                                                      network_optimizer)  # unrolled_model: w -> w'
        # Lval: validation loss
        unrolled_loss = unrolled_model._loss(input_valid, target_valid, self.gumbel)

        unrolled_loss.backward()
        # dαLval(w',α)
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]  # grad wrt alpha

        # dw'Lval(w',α)
        # vector = [v.grad.data for v in unrolled_model.parameters()]  # unrolled_model.parameters(): w‘

        # with adaptive stopping: if a layer is frozen, set its grad to None
        # gumbel-softmax
        if self.gumbel:
            vector = []
            for v in unrolled_model.parameters():
                if v.requires_grad:
                    # not frozen
                    if v.grad is not None:
                        # used operation by Gumbel-softmax
                        vector.append(v.grad.data)
                    else:
                        # unused operation by Gumbel-softmax
                        vector.append(torch.zeros_like(v))
                else:
                    # frozen layers
                    vector.append(None)
        else:
            vector = [v.grad.data if v.requires_grad else None for v in unrolled_model.parameters()]
        
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
            # g.data.sub_(ig.data, alpha=eta)
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
        vector_trim = vector
        # for adaptive stopping: remove the None's, which indicate frozen layers
        if self.adaptive_stop:
            vector_trim = [v for v in vector if v is not None]
        R = r / _concat(vector_trim).norm()

        for p, v in zip(self.model.parameters(), vector):
            # pass the frozen layers
            if v is not None:
                p.data.add_(v, alpha=R)

        loss = self.model._loss(input, target, self.gumbel)
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            # pass the frozen layers
            if v is not None:
                p.data.sub_(v, alpha=2 * R)

        loss = self.model._loss(input, target, self.gumbel)
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            # pass the frozen layers
            if v is not None:
                p.data.add_(v, alpha=R)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
