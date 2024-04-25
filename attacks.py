import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import torchvision
import numpy as np
import skimage
import math
import bisect


class Attack(object):
    '''
    Root class for all adversarial attack classes.
    '''

    def __init__(self, model, targeted=False, img_range=(0, 1)):
        self.model = model
        self.device = next(model.parameters()).device
        self.targeted = targeted
        self.img_range = img_range

    def __repr__(self):
        return str(self.__dict__)

    def to(self, device):
        self.model.to(device)
        self.device = device


##################################### GSE #####################################

class GSEAttack(Attack):
    def __init__(self, model, *args, ver=False, img_range=(-1, 1), search_steps=10,
                 targeted=False, sequential=False, search_factor=2,
                 gb_size=5, sgm=1.5, mu=1, beta=0.0025, iters=200, k_hat=10,
                 q=0.25, **kwargs):
        '''
        Implementation of the GSE attack.

        args:
        model:         Callable, PyTorch classifier.
        ver:           Bool, print progress if True.
        img_range:     Tuple of ints/floats, lower and upper bound of image
                       entries.
        search_steps:  Int, number of steps for line search on the trade-off
                       parameter.
        targeted:      Bool, given label is used as a target label if True.
        sequential:    Bool, perturbations are computed sequentially for all
                       images in the batch if True. For fair comparison to
                       Homotopy attack.
        search_factor: Float, factor to increase/decrease the trade-off
                       parameter until an upper/lower bound for the line search
                       is found.
        gb_size:       Odd int, size of the Gaussian blur kernel.
        sgm:           Float, sigma of the gaussian blur kernel
        mu:            Float, trade-off parameter for 2-norm regularization.
        beta:          Float, step size (sigma in paper)
        iters:         Int, number of iterations.
        k_hat:         Int, number of iterations before transitioning to NAG.
        q:             Float, inverse of increase factor for adjust_lambda.
        '''
        super().__init__(model, img_range=img_range, targeted=targeted)
        self.ver = ver
        self.search_steps = search_steps
        self.sequential = sequential
        self.search_factor = search_factor
        self.gb_size = gb_size
        self.sgm = sgm
        self.mu = mu
        self.beta = beta
        self.iters = iters
        self.k_hat = k_hat
        self.q = q


    def __adjust_lambda(self, lam, noise):
        '''
        Adjust trade-off parameters (lambda) to update search space.
        '''
        x = noise.detach().clone().abs().mean(dim=1, keepdim=True).sign()
        gb = torchvision.transforms.GaussianBlur((self.gb_size, self.gb_size),
                                                 sigma=self.sgm)
        x = gb(x) + 1
        x = torch.where(x == 1, self.q, x)
        lam /= x[:, 0, :, :]
        return lam


    def __section_search(self, x, y, steps=50):
        '''
        Section search for finding the maximal lambda such that the
        perturbation is non-zero after the first iteration.
        '''
        noise = torch.zeros_like(x, requires_grad=True)
        loss = (self.__f(x + noise, y).sum() + self.mu
                * torch.norm(noise, p=2, dim=(1,2,3)).sum())
        loss.backward()
        grad = noise.grad
        noise.detach_()
        ones = torch.ones_like(x)[:, 0, :, :]

        # define upper and lower bound for line search
        lb = torch.zeros((y.size(0),), dtype=torch.float,
                         device=self.device).view(-1, 1, 1)
        ub = lb.clone() + 0.001
        mask = torch.norm(self.__prox(grad.clone() * self.beta,
                                      ones * ub * self.beta),
                          p=0, dim=(1,2,3)) != 0
        while mask.any():
            ub[mask] *= 2
            mask = torch.norm(self.__prox(grad.clone() * self.beta,
                                          ones * ub * self.beta),
                              p=0, dim=(1,2,3)) != 0

        # perform search
        for _ in range(steps):
            cur = (ub + lb) / 2
            mask = torch.norm(self.__prox(grad.clone() * self.beta,
                                          ones * cur * self.beta),
                              p=0, dim=(1,2,3)) == 0
            ub[mask] = cur[mask]
            mask = torch.logical_not(mask)
            lb[mask] = cur[mask]

        return (lb + ub).view(-1) / 2


    def __call__(self, x, y):
        '''
        Call the attack for a batch of images x or sequentially for all images
        in x depending on self.sequential.

        args:
        x:   Tensor of shape [B, C, H, W], batch of images.
        y:   Tensor of shape [B], batch of labels.

        Returns a tensor of the same shape as x containing adversarial examples
        '''
        if self.sequential:
            result = x.clone()
            for i, (x_, y_) in enumerate(zip(x, y)):
                result[i] = self.__perform_att(x_.unsqueeze(0),
                                             y_.unsqueeze(0),
                                             mu=self.mu, beta=self.beta,
                                             k_hat=self.k_hat).detach()
            return result
        else:
            return self.__perform_att(x, y, mu=self.mu, beta=self.beta,
                                    k_hat=self.k_hat)


    def __perform_att(self, x, y, mu, beta, k_hat):
        '''
        Perform GSI attack on a batch of images x with corresponding labels y.
        '''
        x = x.to(self.device)
        y = y.to(self.device)
        B, C, _, _ = x.shape
        lams = self.__section_search(x, y)
        # save x, y, and lams for resetting them at the beginning of every
        # section search step
        save_x = x.clone()
        save_y = y.clone()
        save_lams = lams.clone()
        # upper and lower bounds for section learch
        ub_lams = torch.full_like(lams, torch.inf)
        lb_lams = torch.full_like(lams, 0.0)
        # tensor for saving succesful adversarial examples in inner loop
        result = x.clone()
        # tensor for saving best adversarial example so far
        result2 = x.clone()
        best_l0 = torch.full((B,), torch.inf, device=self.device).type(x.type())

        # section search
        for step in range(self.search_steps):
            x = save_x.clone()
            y = save_y.clone()
            lams = save_lams.clone()
            lam = torch.ones_like(x)[:, 0, :, :] * lams.view(-1, 1, 1)
            # tensor for tracking for which images adv. examples have been found
            active = torch.ones(B, dtype=bool, device=self.device)
            # set initial perturbation to zero
            noise = torch.zeros_like(x, requires_grad = True)
            # epsilon for numerical stability of AdaGrad update
            eps = torch.full_like(x, 1e-9)
            # tensor for accumulating squared gradients for AdaGrad
            G = torch.zeros_like(noise)

            # attack
            for j in range(self.iters):
                if self.ver:
                    print(f'\rSearch step {step + 1}/{self.search_steps}, ' +
                          f'Prox.Grad. Iteration {j + 1}/{self.iters}, ' +
                          f'Images left: {x.shape[0]}', end='')
                if len(x) == 0:
                    break

                self.model.zero_grad()
                loss = (self.__f(x + noise, y).sum() + mu
                        * (torch.norm(noise, p=2, dim=(1,2,3)) ** 2).sum())
                loss.backward()

                with torch.no_grad():
                    # update search space
                    if j % k_hat == 0 or j < 10:
                        noise_ = noise - noise.grad.data
                        noise_ = self.__prox(noise_, lam)
                        lam = self.__adjust_lambda(lam, noise_)
                        lammask = (lam > lams.view(-1, 1, 1))[:, None, :, :]
                        lammask = lammask.repeat(1, C, 1, 1)

                    # AdaGrad update
                    G += noise.grad.data ** 2
                    noise = noise - (beta / torch.sqrt(G + eps)) * noise.grad.data
                    noise[lammask] = 0

                    # clamp adv. example to valid range
                    x_adv = torch.clamp(x + noise, *self.img_range)
                    noise = x_adv - x

                    # save all succesful adv. examples and stop optimizing them
                    preds = torch.argmax(self.model(x_adv), dim=1)
                    mask = preds == y if self.targeted else preds != y
                    if mask.any():
                        tmp = result[active]
                        tmp[mask] = x_adv[mask]
                        result[active] = tmp
                        mask = torch.logical_not(mask)
                        active[active.clone()] = mask
                        x, y, noise, G = x[mask], y[mask], noise[mask], G[mask]
                        lams, lam, lammask = lams[mask], lam[mask], lammask[mask]
                        eps = eps[mask]

                noise.requires_grad = True

            # section search
            # no adv. example found => decrease upper bound and current lambda
            # adv. example found => save it if the "0-norm" is better than of the
            # previous adv. example, increase lower bound and current lambda
            for i in range(B):
                if active[i]:
                    ub_lams[i] = save_lams[i]
                    save_lams[i] = (ub_lams[i] + save_lams[i]) / 2
                else:
                    l0 = (result[i] - save_x[i]).abs().mean(dim=0).norm(p=0)
                    if l0 < best_l0[i]:
                        best_l0[i] = l0
                        result2[i] = result[i].clone()
                    if torch.isinf(ub_lams[i]):
                        lb_lams[i] = save_lams[i]
                        save_lams[i] *= self.search_factor
                    else:
                        lb_lams[i] = save_lams[i]
                        save_lams[i] = (ub_lams[i] + save_lams[i]) / 2

        if self.ver:
            print('')
        return result2.detach()


    def __f(self, x, y, kappa=0):
        '''
        CW loss function
        '''
        logits = self.model(x)
        one_hot_y = F.one_hot(y, logits.size(1))
        Z_t = torch.sum(logits * one_hot_y, dim=1)
        Z_i = torch.amax(logits * (1 - one_hot_y) - (one_hot_y * 1e5), dim=1)
        if self.targeted:
            return F.relu(Z_i - Z_t + kappa)
        else:
            return F.relu(Z_t - Z_i + kappa)


    def __prox(self, grad_loss_noise, lam):
        '''
        Computes the proximal operator of the 1/2-norm of the gradient of the
        adversarial loss wrt current noise.
        '''

        lam = lam[:, None, :, :]
        sh = list(grad_loss_noise.shape)
        lam = lam.expand(*sh)

        p_lam = (54 ** (1 / 3) / 4) * lam ** (2 / 3)

        mask1 = (grad_loss_noise > p_lam)
        mask2 = (torch.abs(grad_loss_noise) <= p_lam)
        mask3 = (grad_loss_noise < -p_lam)
        mask4 = mask1 + mask3

        phi_lam_x = torch.arccos((lam / 8) * (torch.abs(grad_loss_noise) / 3)
                                 ** (-1.5))

        grad_loss_noise[mask4] = ((2 / 3) * torch.abs(grad_loss_noise[mask4])
                                  * (1 + torch.cos((2 * math.pi) / 3
                                  - (2 * phi_lam_x[mask4]) / 3)))
        grad_loss_noise[mask3] = -grad_loss_noise[mask3]
        grad_loss_noise[mask2] = 0

        return grad_loss_noise


################################### Fwnucl ####################################

class FWnucl(Attack):
    def __init__(self, model, *args, iters=200, img_range=(-1, 1), ver=False,
                 targeted=False, eps=5, **kwargs):
        '''
        Implementation of the nuclear group norm attack.

        args:
        model:         Callable, PyTorch classifier.
        ver:           Bool, print progress if True.
        img_range:     Tuple of ints/floats, lower and upper bound of image
                       entries.
        targeted:      Bool, given label is used as a target label if True.
        eps:           Float, radius of the nuclear group norm ball.
        '''
        super().__init__(model, img_range=img_range, targeted=targeted)
        self.iters = iters
        self.ver = ver
        self.eps = eps
        self.gr = (math.sqrt(5) + 1) / 2


    def __loss_fn(self, x, y, lossfn):
        '''
        Compute loss depending on self.targeted.
        '''
        if self.targeted:
            return -lossfn(x, y)
        else:
            return lossfn(x, y)


    def __call__(self, x, y):
        '''
        Perform the nuclear group norm attack on a batch of images x.

        args:
        x:   Tensor of shape [B, C, H, W], batch of images.
        y:   Tensor of shape [B], batch of labels.

        Returns a tensor of the same shape as x containing adversarial examples
        '''
        
        x = x.to(self.device)
        y = y.to(self.device)
        lossfn = nn.CrossEntropyLoss()
        noise = torch.zeros_like(x)
        noise.requires_grad = True

        for t in range(self.iters):
            if self.ver:
                print(f'\rIteration {t+1}/{self.iters}', end='')
            self.model.zero_grad()
            out = self.model(x + noise)
            loss = self.__loss_fn(out, y, lossfn)
            loss.backward()
            s = self.__groupNuclearLMO(noise.grad.data, eps=self.eps)
            with torch.no_grad():
                gamma = self.__lineSearch(x, s, noise, y)
                noise = (1 - gamma) * noise + gamma * s
            noise.requires_grad = True

        x = torch.clamp(x + noise, -1, 1)
        if self.ver:
            print("")
        return x.detach()


    def __lineSearch(self, x, s, noise, y, steps=25):
        '''
        Perform line search for the step size.
        '''
        a = torch.zeros(y.shape, device=self.device).view(-1, 1, 1, 1)
        b = torch.ones(y.shape, device=self.device).view(-1, 1, 1, 1)
        c = b - (b - a) / self.gr
        d = a + (b - a) / self.gr
        lossfn = nn.CrossEntropyLoss(reduction='none')
        sx = s - noise

        for i in range(steps):
            loss1 = self.__loss_fn(self.model(x + noise + c * sx), y, lossfn)
            loss2 = self.__loss_fn(self.model(x + noise + d * sx), y, lossfn)
            mask = loss1 > loss2

            b[mask] = d[mask]
            mask = torch.logical_not(mask)
            a[mask] = c[mask]

            c = b - (b - a) / self.gr
            d = a + (b - a) / self.gr

        return (b + a) / 2


    def __groupNuclearLMO(self, x, eps=5):
        '''
        LMO for the nuclear group norm ball.
        '''

        B, C, H, W = x.shape
        size = 32 if H > 64 else 4

        # turn batch of images into batch of size by size pixel groups per
        # color channel
        xrgb = [x[:, c, :, :] for c in range(C)]
        xrgb = [xc.unfold(1, size, size).unfold(2, size, size) for xc in xrgb]
        xrgb = [xc.reshape(-1, size, size) for xc in xrgb]

        # compute nuclear norm of each patch (sum norms over color channels)
        norms = torch.linalg.svdvals(xrgb[0])
        for xc in xrgb[1:]:
            norms += torch.linalg.svdvals(xc)
        norms = norms.sum(-1).reshape(B, -1)

        # only keep the patch g* with the largest nuclear norm for each image
        idxs = norms.argmax(dim=1).view(-1, 1)
        xrgb = [xc.reshape(B, -1, size, size) for xc in xrgb]
        xrgb = [xc[torch.arange(B).view(-1, 1), idxs].view(B, size, size)
                for xc in xrgb]

        # build index tensor corr. to the position of the kept patches in x
        off = (idxs % (W / size)).long() * size
        off += torch.floor(idxs / (W / size)).long() * W * size
        idxs = torch.arange(0, size**2,
                            device=self.device).view(1, -1).repeat(B, 1) + off
        off = torch.arange(0, size,
                           device=self.device).view(-1, 1).repeat(1, size)
        off = off * W  - off * size
        idxs += off.view(1, -1)

        # compute singular vector pairs corresponding to largest singular value
        # and final perturbation (LMO solution)
        pert = torch.zeros_like(x)
        for i, xc in enumerate(xrgb):
            U, _, V = torch.linalg.svd(xc)
            U = U[:, :, 0].view(B, size, 1)
            V = V.transpose(-2, -1)[:, :, 0].view(B, size, 1)
            pert_gr = torch.bmm(U, V.transpose(-2, -1)).reshape(B, size * size)
            idx = torch.arange(B).view(-1, 1)
            pert_tmp = pert[:, i, :, :].view(B, -1)
            pert_tmp[idx, idxs] = pert_gr * eps
            pert[:, i, :, :] = pert_tmp.view(B, H, W)

        return pert

#################################### SAIF #####################################

class SAIF(Attack):
    def __init__(self, model, *args, targeted=False, img_range=(-1, 1), steps=200,
                 r0=1, ver=False, k=25, eps=1.0, **kwargs):
        '''
        Implementation of the sparse Frank-Wolfe attack SAIF
        https://arxiv.org/pdf/2212.07495.pdf

        args:
        model:         Callable, PyTorch classifier.
        img_range:     Tuple of ints/floats, lower and upper bound of image
                       entries.
        targeted:      Bool, given label is used as a target label if True.
        steps:         Int, number of FW iterations.
        r0:            Int, parameter for step size computation.
        ver:           Bool, print progress if True.
        '''
        super().__init__(model, targeted=targeted, img_range=img_range)
        self.steps = steps
        self.r0 = r0
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.ver = ver
        self.k = k
        self.eps = eps


    def __call__(self, x, y):
        '''
        Perform the  attack on a batch of images x.

        args:
        x:   Tensor of shape [B, C, H, W], batch of images.
        y:   Tensor of shape [B], batch of labels.
        k:   Int, sparsity parameter,
        eps: Float, perturbation magnitude parameter.

        Returns a tensor of the same shape as x containing adversarial examples.
        '''
        B, C, H, W = x.shape
        x = x.to(self.device)
        y = y.to(self.device)
        batchidx = torch.arange(B).view(-1, 1)

        # compute p_0 and s_0
        x_ = x.clone()
        x_.requires_grad = True
        out = self.model(x_)
        loss = -self.loss_fn(out, y)
        loss.backward()
        p = -self.eps * x_.grad.sign()
        p = p.detach()
        ksmallest = torch.topk(-x_.grad.view(B, -1), self.k, dim=1)[1]
        ksmask = torch.zeros((B, C * H * W), device=self.device)
        ksmask[batchidx, ksmallest] = 1
        s = ksmask.view(B, C, H, W).detach().float()

        r = self.r0

        for t in range(self.steps):
            if self.ver:
                print(f'\r Iteration {t+1}/{self.steps}', end='')
            p.requires_grad = True
            s.requires_grad = True

            D = self.__Dtadv(x, s, p, y)
            D.backward()
            mp = p.grad
            ms = s.grad

            with torch.no_grad():
                # inf-norm LMO
                v = - self.eps * mp.sign()

                # 1-norm LMO
                ksmallest = torch.topk(-ms.view(B, -1), self.k, dim=1)[1]
                ksmask = torch.zeros((B, C * H * W), device=self.device)
                ksmask[batchidx, ksmallest] = 1
                ksmask = ksmask.view(B, C, H, W)
                z = torch.logical_and(ksmask, ms < 0).float()

                # update stepsize until primal progress is made
                mu = 1 / (2 ** r * math.sqrt(t + 1))
                while self.__Dtadv(x, s + mu * (z - s), p + mu * (v - p), y) > D:
                    r += 1
                    mu = 1 / (2 ** r * math.sqrt(t + 1))

                p = p + mu * (v - p)
                s = s + mu * (z - s)

                x_adv = torch.clamp(x + p, *self.img_range)
                p = x_adv - x

        if self.ver:
            print('')
        return (x + s * p).detach()


    def __Dtadv(self, x, s, p, y):
        out = self.model(x + s * p)
        if self.targeted:
            return self.loss_fn(out, y)
        else:
            return -self.loss_fn(out, y)

################################## StrAttack ##################################

class StrAttack(Attack):
    def __init__(self, model, *args, targeted=False, img_range=(-1, 1), kappa=0,
                 max_iter=200, ver=False, search_steps=8, max_c=1e10, rho=1,
                 c=2.5, retrain=True, **kwargs):
        '''
        Implementation of StrAttack: https://arxiv.org/abs/1808.01664
        Adapted from https://github.com/KaidiXu/StrAttack

        args:
        model:         Callable, PyTorch classifier.
        targeted:      Bool, given label is used as a target label if True.
        img_range:     Tuple of ints/floats, lower and upper bound of image
                       entries.
        kappa:         Float, confidence parameter for CW loss.
        max_iter:      Int, number of iterations.
        ver:           Bool, print progress if True.
        search_steps:  Int, number of binary search steps.
        max_c:         Float, upper bound for regularizaion parameter.
        rho:           Float, ADMM parameter.
        c:             Float, initial regularization parameter.
        '''
        super().__init__(model, targeted=targeted, img_range=img_range)
        self.kappa = kappa
        self.max_iter = max_iter
        self.ver = ver
        self.search_steps = search_steps
        self.max_c = max_c
        self.rho = rho
        self.c = c
        self.retrain = retrain

    def __compare(self, x, y):
        return torch.equal(x, y) if self.targeted else not torch.equal(x, y)

    def __f(self, x, y):
        '''
        CW loss function
        '''
        logits = self.model(x)
        one_hot_labels = F.one_hot(y, logits.size(1)).to(self.device)
        Z_i = torch.max(logits * (1 - one_hot_labels) - one_hot_labels
                        * 10000., dim=1)[0]
        Z_t = torch.sum(logits * one_hot_labels, dim=1)
        Zdif = Z_i - Z_t if self.targeted else Z_t - Z_i
        return torch.clamp(Zdif + self.kappa, min=0.0)


    def __call__(self, imgs, labs):
        '''
        Perform StrAttack on a batch of images x with corresponding labels y.

        args:
        x:   Tensor of shape [B, C, H, W], batch of images.
        y:   Tensor of shape [B], batch of labels.

        Returns a tensor of the same shape as x containing adversarial examples
        '''
        c_ = self.c
        imgs = imgs.to(self.device)
        labs = labs.to(self.device)
        sh = imgs.shape
        batch_size = sh[0]

        alpha, tau, gamma = 5, 2, 1
        eps = torch.full_like(imgs, 1.0)
        # 8 for imagenet, 2 for CIFAR and MNIST
        filterSize = 8 if sh[-1] > 32 else 2
        stride = filterSize
        # convolution kernel used to compute norm of each group
        slidingM = torch.ones((1, sh[1], filterSize, filterSize), device=self.device)

        cs = torch.ones(batch_size, device=self.device) * c_
        lower_bound = torch.zeros(batch_size)
        upper_bound = torch.ones(batch_size) * self.max_c

        o_bestl2 = torch.full_like(labs, 1e10)
        o_bestscore = torch.full_like(labs, -1)
        o_bestattack = imgs.clone()
        o_besty = torch.ones_like(imgs)

        for step in range(self.search_steps):

            bestl2 = torch.full_like(labs, 1e10)
            bestscore = torch.full_like(labs, -1)

            z, v, u, s = (torch.zeros_like(imgs) for _ in range(4))

            for iter_ in range(self.max_iter):
                if (not iter_%10 or iter_ == self.max_iter - 1) and self.ver:
                    print(f'\rIteration: {iter_+1}/{self.max_iter}, ' +
                          f'Search Step: {step+1}/{self.search_steps}', end='')

                # first update step (7) / Proposition 1
                delta = self.rho / (self.rho + 2 * gamma) * (z - u / self.rho)

                b = z - s / self.rho
                tmp = torch.minimum(self.img_range[1] - imgs, eps)
                w = torch.where(b > tmp, tmp, b)
                tmp = torch.maximum(self.img_range[0] - imgs, -eps)
                w = torch.where(b < tmp, tmp, w)

                c = z - v / self.rho
                cNorm = torch.sqrt(F.conv2d(c ** 2, slidingM, stride=stride))
                cNorm = torch.where(cNorm == 0, torch.full_like(cNorm, 1e-12), cNorm)
                cNorm = F.interpolate(cNorm, scale_factor=filterSize)
                y = torch.clamp((1 - tau / (self.rho * cNorm)), 0) * c

                # second update step (8) / equation (15)
                z_grads = self.__get_z_grad(imgs, labs, z.clone(), cs)
                eta = alpha * math.sqrt(iter_ + 1)
                coeff = (1 / (eta + 3 * self.rho))
                z = coeff * (eta * z + self.rho * (delta + w + y) + u + s + v - z_grads)

                # third update step (9)
                u = u + self.rho * (delta - z)
                v = v + self.rho * (y - z)
                s = s + self.rho * (w - z)

                # get info for binary search
                x = imgs + y
                scores = self.model(x)
                l2s = torch.sum((z ** 2).reshape(z.size(0), -1), dim=-1)

                for i, (l2, sc, x_) in enumerate(zip(l2s, scores, x)):
                    if l2 < bestl2[i] and self.__compare(asc:=torch.argmax(sc), labs[i]):
                        bestl2[i] = l2
                        bestscore[i] = asc
                    if l2 < o_bestl2[i] and self.__compare(asc:=torch.argmax(sc), labs[i]):
                        o_bestl2[i] = l2
                        o_bestscore[i] = asc
                        o_bestattack[i] = x_.detach().clone()
                        o_besty[i] = y[i]

            for i in range(batch_size):
                if (self.__compare(bestscore[i], labs[i]) and bestscore[i] != -1 and bestl2[i] == o_bestl2[i]):
                    upper_bound[i] = min(upper_bound[i], cs[i])
                    if upper_bound[i] < 1e9:
                        cs[i] = (lower_bound[i] + upper_bound[i]) / 2
                else:
                    lower_bound[i] = max(lower_bound[i], cs[i])
                    if upper_bound[i] < 1e9:
                        cs[i] = (lower_bound[i] + upper_bound[i]) / 2
                    else:
                        cs[i] *= 5

        del v, u, s, z_grads, w, tmp

        if self.retrain:
            cs = torch.full_like(labs, 5.0)
            zeros = torch.zeros_like(imgs)

            for step in range(8):
                bestl2 = torch.full_like(labs, 1e10)
                bestscore = torch.full_like(labs, -1)

                Nz = o_besty[o_besty != 0]
                e0 = torch.quantile(Nz.abs(), 0.03)
                A2 = torch.where(o_besty.abs() <= e0, 0, 1)
                z1 = o_besty
                u1 = torch.zeros_like(imgs)
                tmpc = self.rho / (self.rho + gamma / 100)

                for j in range(200):
                    if self.ver and not j % 10:
                        print(f'\rRetrain iteration: {step+1}/8, ' +
                              f'Search Step: {j+1}/200', end='')

                    tmpA = (z1 - u1) * tmpc
                    tmpA1 = torch.where(o_besty.abs() <= e0, zeros, tmpA)
                    cond = torch.logical_and(tmpA >
                                             torch.minimum(self.img_range[1] - imgs, eps),
                                             o_besty.abs() > e0)
                    tmpA2 = torch.where(cond, torch.minimum(self.img_range[1] - imgs, eps),
                                        tmpA1)
                    cond = torch.logical_and(tmpA <
                                             torch.maximum(self.img_range[0] - imgs, -eps),
                                             o_besty.abs() > e0)
                    deltA = torch.where(cond, torch.maximum(self.img_range[0] - imgs, -eps),
                                        tmpA2)

                    x = imgs + deltA
                    scores = self.model(x)
                    l2s = torch.sum((z ** 2).reshape(z.size(0), -1), dim=-1)
                    grad = self.__get_z_grad(imgs, labs, deltA, cs)

                    stepsize = 1 / (alpha + 2 * self.rho)
                    z1 = stepsize * (alpha * z1 * self.rho
                                     * (deltA + u1) - grad * A2)
                    u1 = u1 + deltA - z1

                    for i, (l2, sc, x_) in enumerate(zip(l2s, scores, x)):
                        if (l2 < bestl2[i] and self.__compare(asc:=torch.argmax(sc), labs[i])):
                            bestl2[i] = l2
                            bestscore[i] = asc
                        if (l2 < o_bestl2[i] and self.__compare(asc:=torch.argmax(sc), labs[i])):
                            o_bestl2[i] = l2
                            o_bestscore[i] = asc
                            o_bestattack[i] = x_.detach().clone()
                            o_besty[i] = deltA[i]


                for i in range(batch_size):
                    if self.__compare(bestscore[i], labs[i]) and bestscore[i] != -1 and bestl2[i] == o_bestl2[i]:
                        upper_bound[i] = min(upper_bound[i], cs[i])
                        if upper_bound[i] < 1e9:
                            cs[i] = (lower_bound[i] + upper_bound[i]) / 2
                    else:
                        lower_bound[i] = max(lower_bound[i], cs[i])
                        if upper_bound[i] < 1e9:
                            cs[i] = (lower_bound[i] + upper_bound[i]) / 2
                        else:
                            cs[i] *= 5

        if self.ver:
            print('')
        return o_bestattack


    def __get_z_grad(self, imgs, y, z, cs):
        '''
        Compute and return gradient of loss wrt. z.
        '''
        z.requires_grad = True
        tmp = self.__f(z + imgs, y)
        loss = torch.mean(cs * tmp)
        loss.backward()
        z.detach_()
        return z.grad.data


################################# Homotopy ####################################

class HomotopyAttack(Attack):
    def __init__(self, model, *args, targeted=False, img_range=(-1, 1), ver=False,
                 loss_type='cw', max_epsilon=0.1, dec_factor=0.98, val_c=1e-2,
                 val_w1=1e-1, val_w2=1e-3, max_update=1, maxiter=100,
                 val_gamma=0.8, eta=0.9, delta=0.3, rho=0.8, beta=1e-2,
                 iter_init=50, kappa=0.0, iter_inc=[], n_segments=500, **kwargs):
        '''
        Implementation of group-wise sparse Homotopy attack:
        https://arxiv.org/abs/2106.06027
        Adapted from https://github.com/VITA-Group/SparseADV_Homotopy

        args:
        model:         Callable, PyTorch classifier.
        targeted:      Bool, given label is used as a target label if True.
        img_range:     Tuple of ints/floats, lower and upper bound of image
                       entries.
        ver:           Bool, print progress if True.
        loss_type:     Str 'cw' or 'ce', CW loss or corss entropy loss
        max_epsilon:   Float, upper bound for perturbation magnitude.
        def_factor:    Float, decrease factor for lambda
        val_c:         Float, factor for lambda after initial search.
        val_w1:        Float, trade-off parameter for classification loss.
        val_w2:        Float, trade-off parameter for regularization.
        max_update:    Int, maximum number of super pixels zo be updated per
                       iteration.
        maxiter:       Int, maximum number of nmAPG iterations.
        val_gamma:     Float, constant for criterion (Eq. 12)
        eta:           Float, nmAPG parameter.
        delta:         Float, nmAPG parameter.
        rho:           Float, nmAPG parameter.
        beta:          Float, parameter for increasing lambda during the first
                       search.
        iters_init:    Int, used for computing the number of iterations for the
                       ater attack routine.
        kappa:         Float, confidence parameter for the CW loss.
        iter_inc:      List of ints, used for computing the number of
                       iterations for the ater attack routine.
        n_segments:    Int, initial number of super pixels given to SLIC.
        '''

        super().__init__(model, targeted=targeted, img_range=img_range)
        self.ver = ver
        self.loss_type = loss_type
        self.max_epsilon = max_epsilon
        self.dec_factor = dec_factor
        self.val_c = val_c
        self.val_w1 = val_w1
        self.val_w2 = val_w2
        self.max_update = max_update
        self.maxiter = maxiter
        self.val_gamma = val_gamma
        self.eta = eta
        self.delta = delta
        self.rho = rho
        self.beta = beta
        self.iter_init = iter_init
        self.iter_inc = iter_inc
        self.kappa = kappa
        self.n_segments = n_segments


    def __call__(self, x, y):
        '''
        Performs the attack.

        args:
        x:   Tensor of shape [B, C, H, W], batch of images.
        y:   Tensor of shape [B], batch of labels.

        Returns a tensor of the same shape as x containing adversarial examples
        '''
        result = x.clone().cpu()
        for i, (x_, y_) in enumerate(zip(x, y)):
            if self.ver:
                print(f'Image {i+1}/{x.shape[0]}')
            x_ = x_.unsqueeze(0).to(self.device)
            y_ = y_.unsqueeze(0).to(self.device)
            result[i] += self.homotopy(x_, y_).cpu()[0]

        return result.detach()


    def after_attack(self, x, original_img, target_class, post, iters):

        if post == 1:
            s1 = 1e-3
            s2 = 1e-4
            max_iter = 40000
        else:
            s1 = self.val_w2
            s2 = self.val_w1
            max_iter = iters

        mask = torch.where(torch.abs(x.data) > 0, torch.ones(1).to(self.device),
                           torch.zeros(1).to(self.device))
        pre_x = x.data

        for _ in range(max_iter):

            temp = Variable(x.data, requires_grad=True)
            logist = self.model(temp + original_img.data)
            if self.targeted:
                if self.loss_type == 'ce':
                    ce = torch.nn.CrossEntropyLoss()
                    Loss = ce(logist,torch.ones(1).long().to(self.device)*target_class)
                elif self.loss_type == 'cw':
                    Loss = self.CWLoss(logist, torch.ones(1).long().to(self.device)*target_class)
            else:
                Loss = self.CWLoss(logist, torch.ones(1).long().to(self.device)*target_class)

            self.model.zero_grad()
            if temp.grad is not None:
                temp.grad.data.fill_(0)
            Loss.backward()
            grad = temp.grad


            temp2 = Variable(x.data, requires_grad=True)
            Loss2 = torch.norm(temp2, p=float("inf"))
            self.model.zero_grad()
            if temp2.grad is not None:
                temp2.grad.data.fill_(0)
            Loss2.backward()
            grad2 = temp2.grad

            pre_x = x.data
            if post == 0:
                temp2 = temp2.data - s1*grad2.data*mask - s2*grad.data*mask
            else:
                temp2 = temp2.data - s1*grad2.data*mask

            thres = self.max_epsilon
            temp2 = torch.clamp(temp2.data, -thres, thres)
            temp2 = torch.clamp(original_img.data+temp2.data, *self.img_range)

            x = temp2.data - original_img.data


            logist = self.model(x.data + original_img.data)
            _,pre=torch.max(logist,1)
            if(post == 1):
                if self.targeted:
                    if(pre.item() != target_class):
                        return pre_x
                else:
                    if(pre.item() == target_class):
                        return pre_x

        return x


    def F(self, x, lambda1, original_img, target_class):
        temp = Variable(x.data, requires_grad=False)
        logist = self.model(temp+original_img.data)
        if self.targeted:
            if self.loss_type == 'ce':
                ce = torch.nn.CrossEntropyLoss()
                Loss = ce(logist,torch.ones(1).long().to(self.device)*target_class)
            elif self.loss_type == 'cw':
                Loss = self.CWLoss(logist, torch.ones(1).long().to(self.device)*target_class)
        else:
            Loss = self.CWLoss(logist, torch.ones(1).long().to(self.device)*target_class)
        res = Loss.item() + lambda1*torch.norm(x.data,0).item()
        self.model.zero_grad()
        return res


    def prox_pixel(self, x, alpha, lambda1, original_img):
        '''
        Applies the proximal operator of the group norm to x. Each group
        corresponds to a superpixel in the original image.
        '''
        B, C, H, W = x.shape
        temp_x = x.data * torch.ones_like(x)

        thres = self.max_epsilon
        clamp_x = torch.clamp(temp_x, -thres, thres)

        temp_img = original_img + clamp_x
        temp_img = torch.clamp(temp_img, *self.img_range)
        clamp_x = temp_img.data - original_img.data

        temp_x_norm = torch.zeros((self.groups.max(),), device=x.device, dtype=torch.float)
        pi_x_norm = torch.zeros((self.groups.max(),), device=x.device, dtype=torch.float)
        for i in range(self.groups.max()):
            mask = self.groups == i + 1
            mask = mask[None, :, :].repeat(1, C, 1, 1)
            temp_x_norm[i] = (temp_x[mask] ** 2).sum()
            pi_x_norm[i] = (clamp_x[mask] ** 2).sum()

        val = 1 / (2 * alpha * lambda1)
        cond = 1 + val * pi_x_norm > val * temp_x_norm
        idxs = cond.float().nonzero()
        res = x.clone()
        for i in idxs:
            mask = (self.groups == i.item() + 1)
            mask = mask[None, :, :].repeat(1, C, 1, 1)
            res[mask] *= 0
        return res


    def pert_groups(self, x):
        '''
        Checks which superpixels have non-zero perturbation.
        '''
        C = x.shape[1]
        pert = torch.zeros((self.groups.max(),), device=x.device, dtype=torch.float)
        for i in range(self.groups.max()):
            mask = self.groups == i + 1
            mask = mask[None, :, :].repeat(1, C, 1, 1)
            pert[i] = x[mask].abs().sum()
        return pert.norm(p=0)


    def group_thres(self, x, x0norm, max_update):
        '''
        Sets the perturbation for all superpixels except for the k with the 
        largest 2-norm to zero.
        '''
        B, C, W, H = x.shape
        norms = torch.zeros((self.groups.max(),), device=x.device, dtype=torch.float)
        for i in range(self.groups.max()):
            mask = self.groups == i + 1
            mask = mask[None, :, :].repeat(1, C, 1, 1)
            norms[i] = (x[mask] ** 2).sum()
        _, idx = norms.topk(k=x0norm.int() + max_update)
        res = torch.zeros_like(x)
        for i in idx:
            mask = (self.groups == i.item() + 1)
            mask = mask[None, :, :].repeat(1, C, 1, 1)
            res[mask] = x[mask]
        return res


    def nmAPG(self, x0, original_img, lambda1, search_lambda_inc,
              search_lambda_dec, target_class, max_update, oi=0):

        x0_norm0 = self.pert_groups(x0)

        temp = Variable(x0.data, requires_grad=False)
        logist = self.model(temp+original_img.data)
        if self.targeted:
            if self.loss_type == 'ce':
                ce = torch.nn.CrossEntropyLoss()
                Loss = ce(logist, torch.ones(1).long().to(self.device)*target_class)
            elif self.loss_type == 'cw':
                Loss = self.CWLoss(logist, torch.ones(1).long().to(self.device)*target_class)
        else:
            Loss = self.CWLoss(logist, torch.ones(1).long().to(self.device)*target_class)
        self.model.zero_grad()

        z = x0
        y_pre = torch.zeros(original_img.shape).to(self.device)

        pre_loss = 0
        cur_loss = 0

        counter = 0
        success = 0

        alpha_y = 1e-3
        alpha_x = 1e-3

        alpha_min = 1e-20
        alpha_max = 1e20
        x_pre = x0
        x = x0
        t = 1
        t_pre = 0
        c = Loss + lambda1*torch.norm(x.data,0)
        q = 1
        k = 0
        while True:
            y = x + t_pre/t*(z-x) + (t_pre-1)/t*(x-x_pre)

            if k > 0:
                s = y - y_pre.data

                #gradient of yk
                temp_y = Variable(y.data, requires_grad=True)
                logist_y = self.model(temp_y+original_img.data)
                if self.targeted:
                    if self.loss_type == 'ce':
                        ce = torch.nn.CrossEntropyLoss()
                        Loss_y = ce(logist_y, torch.ones(1).long().to(self.device)*target_class)
                    elif self.loss_type == 'cw':
                        Loss_y = self.CWLoss(logist_y, torch.ones(1).long().to(self.device)*target_class)
                else:
                    Loss_y = self.CWLoss(logist_y, torch.ones(1).long().to(self.device)*target_class)
                self.model.zero_grad()
                if temp_y.grad is not None:
                    temp_y.grad.data.fill_(0)
                Loss_y.backward()
                grad_y = temp_y.grad

                #gradient of yk-1
                temp_y_pre = Variable(y_pre.data, requires_grad=True)
                logist_y_pre = self.model(temp_y_pre+original_img.data)
                if self.targeted:
                    if self.loss_type == 'ce':
                        ce = torch.nn.CrossEntropyLoss()
                        Loss_y_pre = ce(logist_y_pre, torch.ones(1).long().to(self.device)*target_class)
                    elif self.loss_type == 'cw':
                        Loss_y_pre = self.CWLoss(logist_y_pre, torch.ones(1).long().to(self.device)*target_class)
                else:
                    Loss_y_pre = self.CWLoss(logist_y_pre, torch.ones(1).long().to(self.device)*target_class)
                self.model.zero_grad()
                if temp_y_pre.grad is not None:
                    temp_y_pre.grad.data.fill_(0)
                Loss_y_pre.backward()
                grad_y_pre = temp_y_pre.grad

                r = grad_y - grad_y_pre

                #prevent error caused by numerical inaccuracy
                if torch.norm(s,1) < 1e-5:
                    s = torch.ones(1).to(self.device)*1e-5

                if torch.norm(r,1) < 1e-10:
                    r = torch.ones(1).to(self.device)*1e-10

                alpha_y = torch.sum(s*r)/torch.sum(r*r)
                alpha_y = alpha_y.item()

            temp_alpha = alpha_y

            if temp_alpha < alpha_min:
                temp_alpha = alpha_min

            if temp_alpha > alpha_max:
                temp_alpha = alpha_max

            if np.isnan(temp_alpha):
                temp_alpha = alpha_min
            alpha_y = temp_alpha

            count1 = 0
            while True:
                count1 = count1 + 1
                if count1 > 1000:
                    break

                temp_y = Variable(y.data, requires_grad=True)
                logist_y = self.model(temp_y+original_img.data)
                if self.targeted:
                    if self.loss_type == 'ce':
                        ce = torch.nn.CrossEntropyLoss()
                        Loss_y = ce(logist_y,torch.ones(1).long().to(self.device)*target_class)
                    elif self.loss_type == 'cw':
                        Loss_y = self.CWLoss(logist_y, torch.ones(1).long().to(self.device)*target_class)
                else:
                    Loss_y = self.CWLoss(logist_y, torch.ones(1).long().to(self.device)*target_class)
                self.model.zero_grad()
                if temp_y.grad is not None:
                    temp_y.grad.data.fill_(0)
                Loss_y.backward()
                grad_y = temp_y.grad

                z = self.prox_pixel(x=y-alpha_y*grad_y,alpha=alpha_y,
                                    lambda1=lambda1,original_img=original_img)

                #increase lambda
                if(search_lambda_inc == 1):
                    if(torch.norm(z,1) != 0):
                        return 0
                    else:
                        return 1

                #decrease lambda
                if(search_lambda_dec == 1):
                    if(torch.norm(z,1) == 0):
                        return 0
                    else:
                        return lambda1

                alpha_y = alpha_y * self.rho
                cond1 = self.F(z, lambda1, original_img,target_class) <= self.F(y, lambda1, original_img,target_class) - self.delta*(torch.norm(z-y,2)*torch.norm(z-y,2))
                cond2 = self.F(z, lambda1, original_img,target_class) <= c - self.delta*(torch.norm(z-y,2)*torch.norm(z-y,2))

                if(cond1 | cond2):
                    break
            if self.ver:
                print(f'\rHomotopy iteration {oi}, nmAPG iteration {k+1}, norm {y.norm(0)}', end='')
            if self.F(z, lambda1, original_img,target_class) <= c - self.delta*(torch.norm(z-y,2)*torch.norm(z-y,2)):
                x_pre = x
                temp_norm0 = self.pert_groups(z)
                if torch.abs(temp_norm0 - x0_norm0) > max_update:
                    z = self.group_thres(z, x0_norm0, max_update)
                    x = z
                else:
                    x = z
            else:

                if k > 0:
                    s = x - y_pre.data

                    temp_x = Variable(x.data, requires_grad=True)
                    logist_x = self.model(temp_x+original_img.data)
                    if self.targeted:
                        if self.loss_type == 'ce':
                            ce = torch.nn.CrossEntropyLoss()
                            Loss_x = ce(logist_x,torch.ones(1).long().to(self.device)*target_class)
                        elif self.loss_type == 'cw':
                            Loss_x = self.CWLoss(logist_x, torch.ones(1).long().to(self.device)*target_class)
                    else:
                        Loss_x = self.CWLoss(logist_x, torch.ones(1).long().to(self.device)*target_class)
                    self.model.zero_grad()
                    if temp_x.grad is not None:
                        temp_x.grad.data.fill_(0)
                    Loss_x.backward()
                    grad_x = temp_x.grad

                    temp_y_pre = Variable(y_pre.data, requires_grad=True)
                    logist_y_pre = self.model(temp_y_pre+original_img.data)
                    if self.targeted:
                        if self.loss_type == 'ce':
                            ce = torch.nn.CrossEntropyLoss()
                            Loss_y_pre = ce(logist_y_pre,torch.ones(1).long().to(self.device)*target_class)
                        elif self.loss_type == 'cw':
                            Loss_y_pre = self.CWLoss(logist_y_pre, torch.ones(1).long().to(self.device)*target_class)
                    else:
                        Loss_y_pre = self.CWLoss(logist_y_pre, torch.ones(1).long().to(self.device)*target_class)
                    self.model.zero_grad()
                    if temp_y_pre.grad is not None:
                        temp_y_pre.grad.data.fill_(0)
                    Loss_y_pre.backward()
                    grad_y_pre = temp_y_pre.grad

                    r = grad_x - grad_y_pre

                    if torch.norm(s, 1) < 1e-5:
                        s = torch.ones(1).to(self.device) * 1e-5

                    if torch.norm(r,1) < 1e-10:
                        r = torch.ones(1).to(self.device)*1e-10

                    alpha_x = torch.sum(s*r)/torch.sum(r*r)
                    alpha_x = alpha_x.item()

                temp_alpha = alpha_x


                if temp_alpha < alpha_min:
                    temp_alpha = alpha_min

                if temp_alpha > alpha_max:
                    temp_alpha = alpha_max
                if np.isnan(temp_alpha):
                    temp_alpha = alpha_min
                alpha_x = temp_alpha

                count2 = 0
                while True:
                    count2 = count2 + 1

                    if count2 > 10:
                        break

                    temp_x = Variable(x.data, requires_grad=True)
                    logist_x = self.model(temp_x + original_img.data)
                    if self.targeted:
                        if self.loss_type == 'ce':
                            ce = torch.nn.CrossEntropyLoss()
                            Loss_x = ce(logist_x, torch.ones(1).long().to(self.device) * target_class)
                        elif self.loss_type == 'cw':
                            Loss_x = self.CWLoss(logist_x, torch.ones(1).long().to(self.device) * target_class)
                    else:
                        Loss_x = self.CWLoss(logist_x, torch.ones(1).long().to(self.device) * target_class)
                    self.model.zero_grad()
                    if temp_x.grad is not None:
                        temp_x.grad.data.fill_(0)
                    Loss_x.backward()
                    grad_x = temp_x.grad

                    v = self.prox_pixel(x=x-alpha_x*grad_x,alpha=alpha_x,lambda1=lambda1,original_img=original_img)
                    alpha_x = self.rho * alpha_x
                    cond3 = self.F(v, lambda1, original_img,target_class) <= c - self.delta*(torch.norm(v-x,2)*torch.norm(v-x,2))

                    if cond3:
                        break
                    if torch.abs(self.F(v, lambda1, original_img,target_class) - (c - self.delta*(torch.norm(v-x,2)*torch.norm(v-x,2)))) < 1e-3:
                        break


                if self.F(z, lambda1, original_img,target_class) <= self.F(v, lambda1, original_img,target_class):
                    x_pre = x
                    temp_norm0 = self.pert_groups(z)
                    if torch.abs(temp_norm0 - x0_norm0) > max_update:
                        z = self.group_thres(z, x0_norm0, max_update)
                        x = z
                    else:
                        x = z
                else:
                    x_pre = x
                    temp_norm0 = self.pert_groups(v)
                    if torch.abs(temp_norm0 - x0_norm0) > max_update:
                        z = self.group_thres(v, x0_norm0, max_update)
                        x = v
                    else:
                        x = v


            thres = self.max_epsilon
            x = torch.clamp(x.data,-thres,thres)
            temp_img = original_img.data + x.data
            temp_img = torch.clamp(temp_img.data, *self.img_range)
            x = temp_img.data - original_img.data

            y_pre = y.data
            t = (np.sqrt(4*t*t+1)+1)/2
            q = self.eta*q + 1
            c = (self.eta*q*c + self.F(x, lambda1, original_img, target_class))/q

            logist = self.model(x.data+original_img.data)
            _,target=torch.max(logist,1)

            k = k + 1

            pre_loss = cur_loss

            if not self.targeted:
                cur_loss = self.CWLoss(logist.data, torch.ones(1).long().to(self.device)*target_class).item()
            else:
                if self.loss_type == 'cw':
                    cur_loss = self.CWLoss(logist.data, torch.ones(1).long().to(self.device)*target_class).item()
                else:
                    ce = torch.nn.CrossEntropyLoss()
                    cur_loss = ce(logist.data, torch.ones(1).long().to(self.device) * target_class).item()
            self.model.zero_grad()

            #success
            if self.targeted:
                if(target == target_class):
                    success = 1
                    break
            else:
                if((target != target_class)):
                    success = 1
                    break

            if ((success == 0) and (k >= self.maxiter) and (np.abs(pre_loss-cur_loss) < 1e-3) and (counter==1)):
                break

            if((k >= self.maxiter) and (np.abs(pre_loss-cur_loss) < 1e-3)):
                counter = 1

        return x, success


    def lambda_test(self, grad, lam, original_img):
        '''
        Check if initial perturbation is all zero for given lambda.
        '''
        znorm = torch.norm(self.prox_pixel(-1e-3 * grad, 1e-3, lam, original_img), p=1, dim=(1,2,3))
        return 0 if znorm != 0 else 1


    def search_lambda(self, original_img, target_class):
        '''
        Initialize lambda.
        '''

        temp_y = Variable(torch.zeros_like(original_img), requires_grad=True)
        logist_y = self.model(temp_y+original_img.data)
        if self.targeted:
            if self.loss_type == 'ce':
                ce = torch.nn.CrossEntropyLoss()
                Loss_y = ce(logist_y, torch.ones(1).long().to(self.device)*target_class)
            elif self.loss_type == 'cw':
                Loss_y = self.CWLoss(logist_y, torch.ones(1).long().to(self.device)*target_class)
        else:
            Loss_y = self.CWLoss(logist_y, torch.ones(1).long().to(self.device)*target_class)
        self.model.zero_grad()
        if temp_y.grad is not None:
            temp_y.grad.data.fill_(0)
        Loss_y.backward()
        grad = temp_y.grad

        lam = self.beta
        while True:
            if not self.lambda_test(grad, lam, original_img):
                lam += self.beta
            else:
                break
        while True:
            if self.lambda_test(grad, lam, original_img):
                lam *= 0.99
            else:
                break
        
        if self.ver:
            print(f'Lambda = {lam * self.val_c}')
        return lam * self.val_c


    def homotopy(self, original_img, target_class):

        self.groups = torch.from_numpy(skimage.segmentation.slic(original_img.cpu().numpy(),
                                                                 n_segments=self.n_segments,
                                                                 channel_axis=1)).to(self.device)
        lambda1 = self.search_lambda(original_img, target_class)

        x = torch.zeros(original_img.shape).to(self.device)
        pre_norm0 = 0
        cur_norm0 = 0

        max_norm0 = torch.norm(torch.ones(x.shape).to(self.device),0).item()
        outer_iter = 0
        max_update = self.max_update

        while True:
            outer_iter = outer_iter + 1
            x, success = self.nmAPG(x0=x, original_img=original_img, lambda1=lambda1, search_lambda_inc=0,
                               search_lambda_dec=0, target_class=target_class,
                               max_update=max_update, oi=outer_iter)
            max_update = self.max_update
            pre_norm0 = cur_norm0
            cur_norm0 = torch.norm(torch.ones(x.shape).to(self.device)*x.data,0).item()
            cur_norm1 = torch.norm(torch.ones(x.shape).to(self.device) * x.data, 1).item()

            #attack fail
            if(cur_norm0 > max_norm0*0.95 and outer_iter*max_update > max_norm0*0.95):
                break

            iters = 0
            if (cur_norm1 <= cur_norm0 * self.max_epsilon * self.val_gamma):
                max_update = 1
                p = bisect.bisect_left(self.iter_inc, cur_norm0)
                iters = (p + 1) * self.iter_init

            if success == 0:
                x = self.after_attack(x, original_img, target_class, post=0, iters=iters)
                lambda1 = self.dec_factor * lambda1
            else:
                break

            logi = self.model(x.data+original_img.data)
            _,cur_class=torch.max(logi,1)
            if self.targeted:
                if((cur_class == target_class)):
                    break
            else:
                if((cur_class != target_class)):
                    break

        x = self.after_attack(x, original_img, target_class, post=1, iters=iters)
        print('')
        return x


    def CWLoss(self, logits, target):
        target = torch.ones(logits.size(0)).type(torch.float).to(self.device).mul(target.float())
        target_one_hot = Variable(torch.eye(logits.size(1)).type(torch.float).to(self.device)[target.long()])

        real = torch.sum(target_one_hot*logits, 1)
        other = torch.max((1-target_one_hot)*logits - (target_one_hot*10000), 1)[0]
        kappa = torch.zeros_like(other).fill_(self.kappa)

        if self.targeted:
            return torch.sum(torch.max(other-real, kappa))
        else :
            return torch.sum(torch.max(real-other, kappa))

class SAAPF(Attack):
    def __init__(self,
                 model,
                 categories=10,
                 device='cuda:0',  
                 max_iters=10, 
                 lambda_1_max_iters=6, 
                 targeted=True, 
                 ver=True, 
                 img_range=(0, 1), 
                 k=200,
                 lambda_1=1e-3,
                 lambda_2=1e-3,
                 init_lambda_1 = 1e-3,
                 lambda_1_LB=0,
                 lambda_1_UB=100,
                 loss='ce',
                 ):
        """
        Implementation of Sparse Adversarial Attack via Perturbation Factorization 
        https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670035.pdf
        Adapted from https://github.com/wubaoyuan/Sparse-Adversarial-Attack/

        Args:
            model:                  Callable, Pytorch Classifier 
            categories:             Int, number of image labels for the dataset. For CIFAR10, set to 10. For IMAGENET, set to 1000. Defaults to 10.
            max_iters:              Int, maximum number of iterations to iter between G and epsilon. Defaults to 1.
            lambda_1_max_iters:     Int, maximum number of iterations for binary search of lambda_1. Defaults to 6.
            targeted:               Bool, given label is used as a target label if set to True. Defaults to False.
            ver:                    Bool, print Progress if set to True. Defaults to True.
            img_range:              Tuple, lower and upper bound of the image entries. Defaults to (0, 1).
            k:                      Int, maximum number of noised pixels. Defaults to 200.
            lambda_1:               Float. Defaults to 1e-3.
            lambda_2 :              Float. Defaults to 1e-3.
            init_lambda_1:          Float, initial value of lambda_1. Defaults to 1e-3.
            lambda_1_LB:            Int, upper bound for lambda_1. Defaults to 0.
            lambda_1_UB:            Int, lower bound for lambda_1. Defaults to 100.
            loss:                   String, loss function to use, 'cw' for Carlini Wagner Loss and 'ce' for Cross Entropy Loss. Defaults to 'cw'.
        """

        super().__init__(model, targeted, img_range)

        self.categories = categories
        self.init_lambda1 = init_lambda_1
        self.k = k
        self.lambda1 = lambda_1
        self.lambda2 = lambda_2
        self.lambda1_search_times = lambda_1_max_iters
        self.lambda_1_LB = lambda_1_LB
        self.lambda_1_UB = lambda_1_UB
        self.loss = loss
        self.max_iters = max_iters
        self.ver = ver

    def __call__(self, x, y):
        """
        Call the attack for a batch of images x.

        Args:
            x: Tensor of shape [B, C, H, W], batch of images.
            y: Tensor of shape [B], batch of labels.

        Returns:
            Returns a tensor of the same shape as x containing adversarial examples
        """
        return (x + self.saapf(x,y)).detach()

    def saapf(self, x, y):
        """
        Perfroms the attack on the batch of images
        Args:
            x: Tensor of shape [B, C, H, W], batch of images.
            y: Tensor of shape [B], batch of labels.

        Returns:
            Returns a tensor of the same shape as x of element-wise multiplication of epsilon and G 
        """
        torch.backends.cudnn.deterministic=True
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        x_clone = x.clone()
        B_array = []
        
        for image in range(x.shape[0]):

            permuted_img = x_clone[image].permute(1,2,0)
            image_4_mask = np.array(permuted_img.cpu(), dtype=np.uint8)  
            segments = skimage.segmentation.slic(image_4_mask, n_segments=150, compactness=10)

            block_num = max(segments.flatten())+1 - min(segments.flatten())     
            B = np.zeros((block_num, x.shape[1], x.shape[3], x.shape[2]))
            
            for index in range(min(segments.flatten()),max(segments.flatten())+1):
                mask = (segments == index)
                B[index-1,:,mask] = 1

            B = torch.from_numpy(B).float().to(self.device)     
            B_array.append(B)   

        noise_Weight = self.compute_sensitive(x, weight_type='gradient')
        G, epsilon = self.train_adptive(x, y, B_array, noise_Weight)
        return epsilon * G

    def train_adptive(self, x, target, B, noise_Weight):
        """
        Peforms Binary search of lambda_1.
        """
        self.lambda1 = self.init_lambda1
        lambda1_upper_bound = self.lambda_1_UB
        lambda1_lower_bound = self.lambda_1_LB
        
        for search_time in range(1, self.lambda1_search_times+1):

            if self.ver:
                print(f"Binary step {search_time}/{self.lambda1_search_times}")
            results = self.train_sgd_atom(x, target, B, noise_Weight)
            
                
            if search_time < self.lambda1_search_times:
                if results['status'] == True:
                    if self.lambda1 < 0.01*self.init_lambda1:  
                        break
                    # success, divide lambda1 by two
                    lambda1_upper_bound = min(lambda1_upper_bound,self.lambda1)
                    if lambda1_upper_bound < self.lambda_1_UB:
                        self.lambda1 = (lambda1_upper_bound+ lambda1_lower_bound)/2
                else:
                    # failure, either multiply by 10 if no solution found yet
                    # or do binary search with the known upper bound
                    lambda1_lower_bound = max(lambda1_lower_bound, self.lambda1)
                    if lambda1_upper_bound < self.lambda_1_UB:
                        self.lambda1 = (lambda1_upper_bound+ lambda1_lower_bound)/2
                    else:
                        self.lambda1 *= 10     

        return results['G'], results['epsilon'] 

    def train_sgd_atom(self, x, target_label, B, noise_Weight):
        """
        Performs iterative optimization of epsilon and G.
        """
        target_label_tensor=target_label.clone()

        G = torch.ones_like(x, dtype=torch.float32, device=self.device)
        epsilon = torch.zeros_like(x, dtype=torch.float32, device=self.device)

        rho1 = 5e-3
        rho2 = 5e-3
        rho3 = 5e-3
        rho4 = 1e-4
        
        cur_lr_e = 0.1
        cur_lr_g = {'cur_step_g': 0.1, 'cur_rho1': rho1, 'cur_rho2': rho2, 'cur_rho3': rho3,'cur_rho4': rho4}
        
        for mm in range(1,self.max_iters + 1):  
            epsilon, cur_lr_e = self.update_epsilon(x, target_label_tensor, epsilon, G, cur_lr_e, B, noise_Weight, mm, False)
            G, cur_lr_g = self.update_G(x, target_label_tensor, epsilon, G, cur_lr_g, B, noise_Weight, mm)

            if self.ver:
                print(f"{mm}/{self.max_iters}")

        G = (G > 0.5).float()
        # recording results per iteration
        if self.targeted:
            num_correct = (torch.argmax(self.model(x + epsilon * G),dim=1) == target_label)
        else:
            num_correct = (torch.argmax(self.model(x + epsilon * G), dim=1) != target_label)
        num_correct = num_correct.float().mean()

        if num_correct > 0.5:
            results_status=True
        else:
            results_status=False
        
        results = {
            'status': results_status,
            'G' : G.detach(),
            'epsilon' : epsilon.detach(),
        }
        return results

    def update_epsilon(self, x, target_label, epsilon, G, init_lr, B, noise_Weight, out_iter, finetune):
        """
        Method for updating epsilon.
        """
        cur_step = init_lr
        train_epochs = int(2000/2.0) if finetune else 2000
    
        for cur_iter in range(1,train_epochs+1): 
            epsilon.requires_grad = True  
            G.requires_grad = False
            
            images_s = x + torch.mul(epsilon, G) 
            images_s = torch.clamp(images_s,self.img_range[0], self.img_range[1])
            prediction = self.model(images_s)
            
            #loss
            if self.loss == 'ce':
                ce = nn.CrossEntropyLoss()
                if self.targeted:
                    loss = ce(prediction, target_label)
                else:
                    loss = -ce(prediction, target_label)
            elif self.loss == 'cw':    
                one_hot_y = F.one_hot(target_label, prediction.size(1))
                Z_t = torch.sum(prediction * one_hot_y, dim=1)
                Z_i = torch.amax(prediction * (1 - one_hot_y) - (one_hot_y * 1e5), dim=1)
                if self.targeted:
                    loss = F.relu(Z_i - Z_t + 0).mean()
                else:
                    loss =  F.relu(Z_t - Z_i + 0).mean()

            if epsilon.grad is not None:  #the first time there is no grad
                epsilon.grad.data.zero_()
            
            loss.backward(retain_graph=True)
            epsilon_cnn_grad = epsilon.grad
            epsilon_grad = 2*epsilon*G*G*noise_Weight*noise_Weight + self.lambda1 * epsilon_cnn_grad
            epsilon = epsilon - cur_step * epsilon_grad
            epsilon = epsilon.detach()  
            
            
            # updating learning rate
            if cur_iter % 50 == 0:
                cur_step = max(cur_step*0.9, 0.001) 

        return epsilon, cur_step
    
    def update_G(self, x, target_label, epsilon, G, init_params, B, noise_Weight, out_iter):
        """
        Method for updating G.
        """
        cur_step = init_params['cur_step_g']
        cur_rho1 = init_params['cur_rho1']
        cur_rho2 = init_params['cur_rho2']
        cur_rho3 = init_params['cur_rho3']
        cur_rho4 = init_params['cur_rho4']

        # initialize y1, y2 as all 1 matrix, and z1, z2, z4 as all zeros matrix
        y1 = torch.ones_like(G)
        y2 = torch.ones_like(G)
        y3 = torch.ones_like(G)
        z1 = torch.zeros_like(G)
        z2 = torch.zeros_like(G)
        z3 = torch.zeros_like(G)
        z4 = torch.zeros(1, device=self.device)
        ones = torch.ones_like(G)

        for cur_iter in range(1,2000+1): 
            G.requires_grad = True
            epsilon.requires_grad = False
            
            # 1.update y1 & y2
            y1 = torch.clamp((G.detach() + z1/cur_rho1), 0.0, 1.0)
            y2 = self.project_shifted_lp_ball(G.detach() + z2/cur_rho2, 0.5*torch.ones_like(G))      
            
            # 2.update y3
            C=G.detach()+z3/cur_rho3
            for image_index in range(len(B)):   
                BC = C[image_index]*B[image_index]                                       
                n,c,w,h = BC.shape
                Norm = torch.norm(BC.reshape(n, c*w*h), p=2, dim=1).reshape((n,1,1,1))   
                coefficient = 1-self.lambda2/(cur_rho3*Norm)    
                coefficient = torch.clamp(coefficient, min=0)   
                BC = coefficient*BC                           
                y3[image_index] = torch.sum(BC, dim=0)  
            
            # 3.update G
            #cnn_grad_G
            image_s = x + torch.mul(G, epsilon)
            image_s = torch.clamp(image_s, self.img_range[0], self.img_range[1])
            prediction = self.model(image_s)
            
            if self.loss == 'ce':
                ce = nn.CrossEntropyLoss()
                
                if self.targeted:
                    loss = ce(prediction, target_label)   
                else:
                    loss = -ce(prediction, target_label)
            elif self.loss == 'cw':
                one_hot_y = F.one_hot(target_label, prediction.size(1))
                Z_t = torch.sum(prediction * one_hot_y, dim=1)
                Z_i = torch.amax(prediction * (1 - one_hot_y) - (one_hot_y * 1e5), dim=1)
                if self.targeted:
                    loss = F.relu(Z_i - Z_t + 0).mean()
                else:
                    loss =  F.relu(Z_t - Z_i + 0).mean()
            
            if G.grad is not None:  #the first time there is no grad
                G.grad.data.zero_()
            loss.backward()
            cnn_grad_G = G.grad
            
            grad_G = 2*G*epsilon*epsilon*noise_Weight*noise_Weight + self.lambda1*cnn_grad_G \
                    + z1 + z2 + z3+ z4.unsqueeze(1).unsqueeze(2).unsqueeze(3)*ones + cur_rho1*(G-y1) \
                    + cur_rho2*(G-y2)+ cur_rho3*(G-y3) \
                    + cur_rho4*(torch.sum(G,dim=(1,2,3)).unsqueeze(1).unsqueeze(2).unsqueeze(3) - self.k)*ones
            G = G - cur_step*grad_G
            G = G.detach()
            
            # 4.update z1,z2,z3,z4
            z1 = z1 + cur_rho1 * (G.detach() - y1)
            z2 = z2 + cur_rho2 * (G.detach() - y2)
            z3 = z3 + cur_rho3 * (G.detach() - y3)
            z4 = z4 + cur_rho4 * (torch.sum(G,dim=(1,2,3))-self.k)

            # 5.updating rho1, rho2, rho3, rho4
            if cur_iter % 1 == 0:
                cur_rho1 = min(1.01 * cur_rho1, 20)
                cur_rho2 = min(1.01 * cur_rho2, 20)
                cur_rho3 = min(1.01 * cur_rho3, 100)
                cur_rho4 = min(1.01 * cur_rho4, 0.01)
                
            # updating learning rate
            if cur_iter % 50 == 0:
                cur_step = max(cur_step*0.9, 0.001)

            cur_iter = cur_iter + 1

        res_param = {'cur_step_g': cur_step, 'cur_rho1': cur_rho1,'cur_rho2': cur_rho2, 'cur_rho3': cur_rho3,'cur_rho4': cur_rho4}
        return G, res_param

    
    def project_shifted_lp_ball(self, x, shift_vec):
        """
        Performs projection onto the box constraint S_b.
        """
        shift_x = x - shift_vec

        # compute L2 norm: sum(abs(v)^2)^(1/2)
        norm2_shift = torch.norm(shift_x, 2, dim=(1,2,3))

        n = float(x.numel())
        xp = (n**(1/2))/2 * (shift_x / norm2_shift.unsqueeze(1).unsqueeze(2).unsqueeze(3)) + shift_vec
        return xp

    def compute_sensitive(self, x, weight_type='none'):
        """
        Computes sensitive.
        """
        weight = torch.ones_like(x) 
        n, c, h, w = x.shape   #1,3,299,299
        
        if weight_type == 'none':
            return weight
            
        else:
            if weight_type == 'gradient':
                from scipy.ndimage import filters

                weight = torch.zeros_like(x)
                for image_index in range(x.shape[0]):
                    im = x[image_index].permute(1,2,0)                            #229,229,3
                    im_Prewitt_x = torch.zeros(im.shape ,dtype=torch.float32, device=self.device)
                    im_Prewitt_y = torch.zeros(im.shape ,dtype=torch.float32, device=self.device)
                    im_Prewitt_xy = torch.zeros(im.shape ,dtype=torch.float32, device=self.device)
            
                    filters.prewitt(im.cpu().numpy(), 1, im_Prewitt_x.cpu().numpy())
                    filters.prewitt(im.cpu().numpy(), 0, im_Prewitt_y.cpu().numpy())
                    im_Prewitt_xy = torch.sqrt(im_Prewitt_x ** 2 + im_Prewitt_y ** 2) 
                    
                    im_Prewitt_xy = im_Prewitt_xy.permute(2,0,1)
                    weight[image_index] = im_Prewitt_xy.clone().float()
        
            else:
                for i in range(h):
                    for j in range(w):
                        left = max(j - 1, 0)
                        right = min(j + 2, w)
                        up = max(i - 1, 0)
                        down = min(i + 2, h)
                        
                        for k in range(c):
                            if weight_type == 'variance':
                                weight[0, k, i, j] = torch.std(x[0, k, up:down, left:right])
                            elif weight_type == 'variance_mean':
                                weight[0, k, i, j] = torch.std(x[0, k, up:down, left:right]) * torch.mean(x[0, k, up:down, left:right])
                            elif weight_type == 'contrast':
                                weight[0, k, i, j] = (torch.max(x[0, k, up:down, left:right]) - torch.min(x[0, k, up:down, left:right])) / (torch.max(x[0, k, up:down, left:right]) + torch.min(x[0, k, up:down, left:right]))
                            elif weight_type == 'contrast_mean':
                                contrast = (torch.max(x[0, k, up:down, left:right]) - torch.min(x[0, k, up:down, left:right])) / (torch.max(x[0, k, up:down, left:right]) + torch.min(x[0, k, up:down, left:right]))
                                weight[0, k, i, j] = contrast * torch.mean(x[0, k, up:down, left:right])
                                
                            if torch.isnan(weight[0, k, i, j]):
                                weight[0, k, i, j] = 1e-4
                                
            weight = 1.0 / (weight + 1e-4) 
            for k in range(c):
                weight[:, k, :, :] = (weight[:, k, :, :] - torch.min(weight[:, k, :, :])) / (torch.max(weight[:, k, :, :]) - torch.min(weight[:, k, :, :]))
            
            return weight