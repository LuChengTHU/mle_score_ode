from copy import copy
import os
import math
import numpy.matlib
import scipy.special

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm

import sklearn
import sklearn.datasets

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

import utils_toy


LOW, HIGH = -1., 1.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# Make some data; a 1D random walk + small fraction of sine waves
time_steps = 1000
mus = torch.from_numpy(np.array([-2., -6., 4.])).to(device).reshape((-1, 1)).float() / 7.
stds = torch.from_numpy(np.array([0.1, 0.1, 1.])).to(device).reshape((-1, 1)).float() / 7. ** 2
probs = torch.from_numpy(np.array([0.3, 0.3, 0.4])).to(device).reshape((-1, 1)).float()

num_series = mus.shape[0] * 1000
T = 1.
beta_0 = 0.1
beta_1 = 20.
sigma_0 = 0.01
sigma_1 = 50.
START_TIME=1e-5

def inf_train_gen(data='mog', dim=1, batch_size=5000):
    if data == 'mog':
        assert batch_size % 10 == 0
        dim = int(dim)
        x = torch.cat([torch.randn(int(batch_size * probs[i]), dim).to(device) * torch.sqrt(stds[i]) + mus[i] for i in range(len(mus))], axis=0).reshape((-1, dim))
        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)
        return x[indices].float()
    elif data == 'point':
        assert dim == 1
        p = [0.2, 0.2, 0.2, 0.2, 0.2]
        points = [-6., -3, 0., 3, 6.]
        length = 0.001
        def uniform(shape, middle, length):
            return torch.rand(shape).to(device) * length + middle - length / 2.
        x = torch.cat([uniform((int(batch_size * p[i]), dim), points[i], length) for i in range(len(points))], axis=0).reshape((-1, dim))
        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)
        return x[indices].float()
    elif data == 'checkerboard':
        assert dim == 2
        x1 = np.random.rand(batch_size) * 4 - 2
        x2_ = np.random.rand(batch_size) - np.random.randint(0, 2, batch_size) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        return torch.from_numpy(np.concatenate([x1[:, None], x2[:, None]], 1) * 2).float()
    elif data == "swissroll":
        assert dim == 2
        data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 10.
        data = torch.from_numpy(data).float()
        r = 4.5
        data1 = data.clone() + torch.tensor([-r, -r])
        data2 = data.clone() + torch.tensor([-r, r])
        data3 = data.clone() + torch.tensor([r, -r])
        data4 = data.clone() + torch.tensor([r, r])
        data = torch.cat([data, data1, data2, data3, data4], axis=0)
        return data


def sigma_fn(t):
    sigma = sigma_0 * (sigma_1 / sigma_0) ** t
    return sigma * np.sqrt(2 * (np.log(sigma_1) - np.log(sigma_0)))

def marginal_mean_fn(x0, t):
    if SDE_TYPE == 'VE':
        return x0
    else:
        return sqrt_one_minus_square_alpha_fn(t).reshape((-1, 1)) * x0

def marginal_std_fn(t):
    if SDE_TYPE == 'VE':
        return sigma_0 * (sigma_1 / sigma_0) ** t
    else:
        return alpha_fn(t)

def g_square_fn(t):
    if SDE_TYPE == 'VE':
        return sigma_fn(t) ** 2
    else:
        return beta_fn(t)

def beta_fn(t):
    return beta_0 + t * (beta_1 - beta_0)

def alpha_fn(t):
    exp_term = beta_0 * t + 0.5 * t * t * (beta_1 - beta_0)
    return torch.where(exp_term <= 1e-3, torch.sqrt(exp_term), torch.sqrt(1. - torch.exp(-exp_term)))

def square_alpha_fn(t):
    exp_term = beta_0 * t + 0.5 * t * t * (beta_1 - beta_0)
    return torch.where(exp_term <= 1e-3, exp_term, 1. - torch.exp(-exp_term))

def sqrt_one_minus_square_alpha_fn(t):
    return torch.sqrt(1. - square_alpha_fn(t))

def mean(t):
    if SDE_TYPE == 'VE':
        return mus
    else:
        return (sqrt_one_minus_square_alpha_fn(t) * mus)

def square_std(t):
    if SDE_TYPE == 'VE':
        return stds + marginal_std_fn(t) ** 2
    else:
        return ((1 - square_alpha_fn(t)) * stds + square_alpha_fn(t))

def ode_func(x, t, score_fn):
    if SDE_TYPE == 'VE':
        return -0.5 * sigma_fn(t)**2 * score_fn(x, t)
    else:
        return (-0.5 * beta_fn(t) * x - 0.5 * beta_fn(t) * score_fn(x, t))

def ode_func_div(x, t, second_score_fn):
    if SDE_TYPE == 'VE':
        return -0.5 * sigma_fn(t)**2 * second_score_fn(x, t)
    else:
        return -0.5 * beta_fn(t) - 0.5 * beta_fn(t) * second_score_fn(x, t)

def ode_func_jvp(x, t, score_jac_fn, v):
    if SDE_TYPE == 'VE':
        return -0.5 * sigma_fn(t)**2 * torch.bmm(score_jac_fn(x, t), v.reshape((x.shape[0], x.shape[1], 1))).reshape((x.shape[0], x.shape[1]))
    else:
        return -0.5 * beta_fn(t) * v - 0.5 * beta_fn(t) * torch.bmm(score_jac_fn(x, t), v.reshape((x.shape[0], x.shape[1], 1))).reshape((x.shape[0], x.shape[1]))

def ode_func_third(x, t, third_score_fn):
    if SDE_TYPE == 'VE':
        return -0.5 * sigma_fn(t)**2 * third_score_fn(x, t)
    else:
        return (-0.5 * beta_fn(t) - 0.5 * beta_fn(t) * third_score_fn(x, t))

def ode_update_forward(x, t, score_fn):
    return ode_func(x, t, score_fn) / time_steps

def ode_update_reverse(x, t, score_fn):
    t = T - t
    return ode_func(x, t, score_fn) / time_steps

def sde_update_forward(x, t):
    if SDE_TYPE == 'VE':
        return sigma_fn(t) * torch.randn(x.shape).to(device) / np.sqrt(time_steps)
    else:
        return (-0.5 * beta_fn(t) * x) / time_steps + torch.sqrt(beta_fn(t)) * torch.randn(x.shape).to(device) / np.sqrt(time_steps)

def sde_update_reverse(x, t, score_fn):
    t = T - t
    if SDE_TYPE == 'VE':
        return -sigma_fn(t)**2 * score_fn(x, t) / time_steps + sigma_fn(t) * torch.randn(x.shape).to(device) / np.sqrt(time_steps)
    else:
        return (-0.5 * beta_fn(t) * x - beta_fn(t) * score_fn(x, t)) / time_steps + torch.sqrt(beta_fn(t)) * torch.randn(x.shape).to(device) / np.sqrt(time_steps)

def ddpm_sample(x, t, score_fn):
    t = T - t
    beta = beta_fn(t) / time_steps
    return (x + beta * score_fn(x, t)) / torch.sqrt(1 - beta) + torch.sqrt(beta) * torch.randn((x.shape[0])).to(device)

def prior_logp(z):
    if SDE_TYPE == 'VE':
        logZ = -0.5 * np.log(2 * np.pi * sigma_1**2)
        return (logZ - z.pow(2) / (2 * sigma_1**2)).sum(dim=1, keepdim=True)
    else:
        logZ = -0.5 * np.log(2 * np.pi)
        return (logZ - z.pow(2) / 2).sum(dim=1, keepdim=True)

def prior_score(z):
    if SDE_TYPE == 'VE':
        return -z / sigma_1**2
    else:
        return -z

def forward_log_q(x, t):
    shape = x.shape
    if shape[1] == 1:
        shape = (1, -1)
    else:
        shape = (1, *shape)
    x = x.reshape(shape)
    t = t.reshape((1, -1))
    gaussian_unnormalized_q = torch.exp(-0.5 * ((x - mean(t))**2) / square_std(t))
    mog_coeff = probs.reshape((-1, 1)) / torch.sqrt(square_std(t))
    q = torch.sum(gaussian_unnormalized_q * mog_coeff, axis=0) / np.sqrt(2 * np.pi)
    return torch.log(q)

def forward_score(x, t):
    def gaussian_prob(x, t):
        return torch.exp(-0.5 * ((x - mean(t))**2) / square_std(t)) / torch.sqrt(2 * np.pi * square_std(t))
    
    def gaussian_score(x, t):
        return -(x- mean(t)) / square_std(t)

    shape = x.shape
    if shape[1] == 1:
        shape = (1, -1)
    else:
        shape = (1, *shape)
    x = x.reshape(shape)
    t = t.reshape((1, -1))
    gaussian_q = gaussian_prob(x, t)
    mog_coeff = probs.reshape((-1, 1))
    q = torch.sum(gaussian_q * mog_coeff, axis=0).reshape(shape)
    score_coeff = mog_coeff * gaussian_q / q
    return torch.sum(score_coeff * gaussian_score(x, t), axis=0).reshape((-1, 1))

def forward_second_score(x, t):
    def gaussian_prob(x, t):
        return torch.exp(-0.5 * ((x - mean(t))**2) / square_std(t)) / torch.sqrt(2 * np.pi * square_std(t))
    
    def gaussian_score(x, t):
        return -(x- mean(t)) / square_std(t)

    def gaussian_second_score(t):
        return -1. / square_std(t)

    shape = x.shape
    if shape[1] == 1:
        shape = (1, -1)
    else:
        shape = (1, *shape)
    x = x.reshape(shape)
    t = t.reshape((1, -1))
    gaussian_q = gaussian_prob(x, t)
    mog_coeff = probs.reshape((-1, 1))
    q = torch.sum(gaussian_q * mog_coeff, axis=0).reshape(shape)
    score_coeff = mog_coeff * gaussian_q / q
    score_div = torch.sum(score_coeff * (gaussian_score(x, t)**2 + gaussian_second_score(t)), axis=0) - (torch.sum(score_coeff * gaussian_score(x, t), axis=0))**2
    return score_div.reshape((-1, 1))

def forward_score_jac(x, t):
    return forward_second_score(x, t).reshape((-1, 1, 1))

def forward_third_score(x, t):
    x.requires_grad_(True)
    second_score = forward_second_score(x, t)
    third_score = torch.autograd.grad(second_score.sum(), x)[0]
    x.requires_grad_(False)
    return third_score.reshape((x.shape[0], 1))

def draw_forward(ax, fig, method, score_fn, eps=START_TIME):
    x0 = torch.cat([torch.randn(int(num_series * probs[i])).to(device) * torch.sqrt(stds[i]) + mus[i] for i in range(len(mus))]).to(device).reshape((-1, 1))
    Y = torch.zeros((num_series, time_steps)).to(device)
    xt = x0
    for t in range(time_steps):
        Y[:,t] = xt.flatten()
        t_real = eps + t / time_steps * (T - eps)
        t_array = torch.ones_like(xt).to(xt) * t_real
        if method == 'ode':
            xt = xt + ode_update_forward(xt, t_array, score_fn)
        else:
            xt = xt + sde_update_forward(xt, t_array)

    # Linearly interpolate between the points in each time series
    num_fine = 800
    xt = xt.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    x = np.linspace(0, 1., time_steps)
    x_fine = np.linspace(x.min(), x.max(), num_fine)
    y_fine = np.empty((num_series, num_fine), dtype=float)
    for i in range(num_series):
        y_fine[i, :] = np.interp(x_fine, x, Y[i, :])
    y_fine = y_fine.flatten()
    y_fine = np.nan_to_num(y_fine, nan=0.)
    x_fine = np.matlib.repmat(x_fine, num_series, 1).flatten()

    # Plot (x, y) points in 2d histogram with log colorscale
    # It is pretty evident that there is some kind of structure under the noise
    # You can tune vmax to make signal more visible
    cmap = copy(plt.cm.plasma)
    cmap.set_bad(cmap(0))
    h, xedges, yedges = np.histogram2d(x_fine, y_fine, bins=[800, 200])
    pcm = ax.pcolormesh(xedges, yedges, h.T, cmap=cmap,
                            norm=LogNorm(vmax=100), rasterized=True)
    fig.colorbar(pcm, ax=ax, label="# points", pad=0)
    ax.set_title("forward process")
    if SDE_TYPE == 'VE':
        ax.set_ylim(LOW * 3 * sigma_1, HIGH * 3 * sigma_1)
    else:
        ax.set_ylim(LOW * 3, HIGH * 3)
    return xt

def draw_reverse(ax, fig, method, score_fn, start=None, title="", eps=START_TIME):
    if start is None:
        if SDE_TYPE == 'VE':
            x0 = torch.cat([torch.randn(int(num_series * probs[i])).to(device) for i in range(len(mus))]).to(device) * sigma_1
        else:
            x0 = torch.cat([torch.randn(int(num_series * probs[i])).to(device) for i in range(len(mus))]).to(device)
    else:
        x0 = start
    x0 = x0.reshape((-1, 1))
    Y = torch.zeros((num_series, time_steps)).to(device)
    xt = x0
    for t in range(time_steps):
        Y[:,time_steps - 1 - t] = xt.flatten()
        t_real = eps + t / time_steps * (T - eps)
        t_array = torch.ones_like(xt).to(xt) * t_real
        if method == 'ode':
            xt = xt - ode_update_reverse(xt, t_array, score_fn)
        elif method == 'sde':
            xt = xt - sde_update_reverse(xt, t_array, score_fn)
        else:
            xt = ddpm_sample(xt, t_array, score_fn)

    # Linearly interpolate between the points in each time series
    num_fine = 800
    xt = xt.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    x = np.linspace(0, 1., time_steps)
    x_fine = np.linspace(x.min(), x.max(), num_fine)
    y_fine = np.empty((num_series, num_fine), dtype=float)
    for i in range(num_series):
        y_fine[i, :] = np.interp(x_fine, x, Y[i, :])
    y_fine = y_fine.flatten()
    y_fine = np.nan_to_num(y_fine, nan=0.)
    x_fine = np.matlib.repmat(x_fine, num_series, 1).flatten()

    # Plot (x, y) points in 2d histogram with log colorscale
    # It is pretty evident that there is some kind of structure under the noise
    # You can tune vmax to make signal more visible
    cmap = copy(plt.cm.plasma)
    cmap.set_bad(cmap(0))
    h, xedges, yedges = np.histogram2d(x_fine, y_fine, bins=[800, 200])
    pcm = ax.pcolormesh(xedges, yedges, h.T, cmap=cmap,
                            norm=LogNorm(vmax=100), rasterized=True)
    fig.colorbar(pcm, ax=ax, label="# points", pad=0)
    ax.set_title("reverse process {}".format(title))
    if SDE_TYPE == 'VE':
        ax.set_ylim(LOW * 3 * sigma_1, HIGH * 3 * sigma_1)
    else:
        ax.set_ylim(LOW * 3, HIGH * 3)

def plt_marginal_density(score_fn, second_score_fn, ax, npts=1000, memory=5000, title="q0(x)", eps=START_TIME, dim=1, LOW=-7., HIGH=7.):
    if dim == 1:
        yy = torch.from_numpy(np.linspace(LOW, HIGH, npts)).to(device).reshape((-1, 1)).float()
        logpx = ode_likelihood(score_fn, second_score_fn, yy, eps)
        ax.plot(yy.flatten().detach().cpu().numpy(), np.exp(logpx.reshape((-1,))))
        ax.grid()
        ax.set_title(title)
    elif dim == 2:
        side = np.linspace(LOW, HIGH, npts)
        xx, yy = np.meshgrid(side, side)
        x = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])
        xt = torch.from_numpy(x).type(torch.float32).to(device)
        inds = torch.arange(0, xt.shape[0]).to(torch.int64)
        logpx = []
        for ii in tqdm(torch.split(inds, memory)):
            logpx.append(ode_likelihood(score_fn, second_score_fn, xt[ii], eps).reshape((-1,)))
        logpx = np.concatenate(logpx, axis=0)
        px = np.exp(logpx).reshape(npts, npts)
        ax.imshow(px, cmap='inferno')
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.set_title(title)
        return px

def plt_marginal_score(score_fn, second_score_fn, score_grad_div_fn, ax, npts=1000, memory=5000, title="q0(x)", eps=START_TIME, is_data=False, LOW=-7., HIGH=7.):
    if is_data:
        yy = torch.from_numpy(np.linspace(LOW, HIGH, npts)).to(device).reshape((-1, 1)).float()
        score_data = forward_score(yy, torch.ones_like(yy).to(yy) * eps).detach().cpu().numpy()
        ax.plot(yy.flatten().detach().cpu().numpy(), score_data.reshape((-1,)))
        ax.grid()
        ax.set_title(title)
        ax.set_ylim(-130, 130)
    else:
        yy = torch.from_numpy(np.linspace(LOW, HIGH, npts)).to(device).reshape((-1, 1)).float()
        score_ode = ode_score(score_fn, second_score_fn, score_grad_div_fn, yy, eps).detach().cpu().numpy()
        ax.plot(yy.flatten().detach().cpu().numpy(), score_ode.reshape((-1,)))
        ax.grid()
        ax.set_title(title)
        ax.set_ylim(-130, 130)

def plt_samples(samples, ax, npts=100, title="$x ~ p(x)$"):
    ax.hist2d(samples[:, 0], samples[:, 1], range=[[LOW, HIGH], [LOW, HIGH]], bins=npts, cmap='inferno')
    ax.invert_yaxis()
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_title(title)

def plt_marginal_samples(score_fn, ax, npts=100, memory=5000, title="q0(x)", eps=START_TIME):
    z = torch.randn(npts * npts, 2).type(torch.float32).to(device)
    zk = []
    inds = torch.arange(0, z.shape[0]).to(torch.int64)
    for ii in torch.split(inds, int(memory)):
        xt = z[ii]
        for t in tqdm(range(time_steps)):
            t_real = eps + t / time_steps * (T - eps)
            t_array = (torch.ones((int(memory), 1)).to(device) * t_real).requires_grad_(False)
            with torch.no_grad():
                xt = xt - ode_update_reverse(xt, t_array, score_fn)
        zk.append(xt)
    zk = torch.cat(zk, 0).cpu().numpy()
    ax.hist2d(zk[:, 0], zk[:, 1], range=[[-4., 4.], [-4., 4.]], bins=npts, cmap='inferno')
    ax.invert_yaxis()
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_title(title)

def plt_flow_density(log_density, ax, npts=400, memory=400, title="q(x)", eps=START_TIME):
    yy = torch.from_numpy(np.linspace(HIGH, LOW, npts)).to(device)
    px = torch.zeros((npts, time_steps)).to(device)
    for t in tqdm(range(time_steps)):
        t_real = eps + t / time_steps * (T - eps)
        t_array = torch.ones_like(yy).to(yy) * t_real
        px[:,t] = torch.exp(log_density(yy, t_array))
    ax.imshow(px.cpu().numpy(), cmap='inferno')
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_title(title)

def plt_score(score_fn, ax, npts=400, memory=400, title="q(x)", predict_eps=False, eps_square=False, eps=START_TIME):
    px = torch.zeros((npts, time_steps)).to(device).requires_grad_(False)
    for t in tqdm(range(time_steps)):
        yy = torch.from_numpy(np.linspace(HIGH, LOW, npts)).to(device).reshape((-1, 1)).float()
        t_real = eps + t / time_steps * (T - eps)
        t_array = torch.ones_like(yy).to(yy) * t_real
        if predict_eps:
            if eps_square:
                px[:,t] = ((alpha_fn(t_array) ** 2) * score_fn(yy, t_array)).flatten()
            else:
                px[:,t] = (alpha_fn(t_array) * score_fn(yy, t_array)).flatten()
        else:
            px[:,t] = (score_fn(yy, t_array)).flatten().detach()
        del yy
        del t_array
    print(px.max(), px.min())
    ax.imshow(px.cpu().numpy(), cmap='inferno')
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_title(title)

def plt_score_trajectory(ax, npts=400, memory=400, title="q(x)", eps=START_TIME, type=''):
    xx = np.linspace(0., 1., time_steps)
    nums = num_series
    x0 = np.concatenate([np.random.randn(int(nums * probs[i])) * np.sqrt(stds[i]) + mus[i] for i in range(len(mus))])
    div = np.zeros((nums, time_steps))
    xt = x0
    for t in range(time_steps):
        # div[:, t] = (-0.5 * beta_ts[t] - 0.5 * beta_ts[t] * forward_score(xt, t / time_steps))
        # div[:, t] = alpha_fn(t / time_steps) * forward_score(xt, t / time_steps)
        # div[:, t] = (-0.5 * beta_ts[t] - 0.5 * beta_ts[t] * forward_second_score(xt, t / time_steps))
        div[:, t] = alpha_fn(t / time_steps) ** 2 * forward_second_score(xt, t / time_steps)
        xt = xt + ode_update_forward(xt, t)
    for n in range(nums):
        ax.plot(xx, div[n,:])
    ax.plot(xx, np.mean(div, axis=0), linestyle='-.')
    ax.set_title(title)

def plt_importance_proposal_density(axes, title="proposal", eps=1e-3):
    exponent1 = 0.5 * eps * (eps - 2) * beta_0 - 0.5 * eps ** 2 * beta_1
    exponent2 = 0.5 * T * (T - 2) * beta_0 - 0.5 * T ** 2 * beta_1
    term1 = np.where(np.abs(exponent1) <= 1e-3, -exponent1, 1. - np.exp(exponent1))
    term2 = np.where(np.abs(exponent2) <= 1e-3, -exponent2, 1. - np.exp(exponent2))
    Z1 = np.log(term2) - exponent2 - np.log(term1) + exponent1
    xx = np.linspace(eps, T, time_steps)
    exponent = 0.5 * xx * (xx - 2) * beta_0 - 0.5 * xx ** 2 * beta_1
    term = np.where(np.abs(exponent) <= 1e-3, -exponent, 1. - np.exp(exponent))
    proposal_1 = beta_fn(xx) / term / Z1
    axes[0].plot(xx, proposal_1)

def ode_likelihood(score_fn, score_div_fn, data, t_start, rtol=1e-5, atol=1e-5, method='RK45'):
    shape = data.shape

    def ode_solver_func(t, x):
        sample = torch.from_numpy(x[:-shape[0]].reshape((shape[0], shape[1]))).to(device).float()
        t_array = (torch.ones((shape[0], 1)).to(device) * t).requires_grad_(False)
        drift = ode_func(sample, t_array, score_fn).detach()
        logp_grad = ode_func_div(sample, t_array, score_div_fn).detach()
        return torch.cat([drift.reshape((-1,)), logp_grad.reshape((-1,))], axis=0).cpu().numpy()

    init = torch.cat([data.reshape((-1,)).cpu(), torch.zeros((shape[0],))], axis=0).reshape((-1,)).numpy()
    solution = integrate.solve_ivp(ode_solver_func, (t_start, T), init, rtol=rtol, atol=atol, method=method)
    nfe = solution.nfev
    t = solution.t
    zp = solution.y[:, -1]
    z = zp[:-shape[0]].reshape(shape)
    delta_logp = zp[-shape[0]:].reshape((shape[0], 1))
    logpz = prior_logp(torch.from_numpy(z)).numpy()
    logpx = logpz + delta_logp
    return logpx

def ode_score(score_fn, score_jac_fn, score_grad_div_fn, data, t_start, rtol=1e-5, atol=1e-5, method='RK45'):
    shape = data.shape

    def ode_solver_func_forward(t, x):
        sample = torch.from_numpy(x.reshape((shape[0], shape[1]))).to(device).float()
        t_array = (torch.ones((shape[0], 1)).to(device) * t).requires_grad_(False)
        drift = ode_func(sample, t_array, score_fn).detach()
        return drift.reshape((-1,)).cpu().numpy() 

    def ode_solver_func_reverse(t, x):
        sample = torch.from_numpy(x[:shape[0]*shape[1]].reshape((shape[0], shape[1]))).to(device).float()
        score = torch.from_numpy(x[shape[0]*shape[1]:].reshape((shape[0], shape[1]))).to(device).float()
        t_array = (torch.ones((shape[0], 1)).to(device) * t).requires_grad_(False)
        drift = ode_func(sample, t_array, score_fn).detach()
        delta_score = -(ode_func_third(sample, t_array, score_grad_div_fn).detach() + ode_func_jvp(sample, t_array, score_jac_fn, score).detach())
        return torch.cat([drift.reshape((-1,)), delta_score.reshape((-1,))], axis=0).cpu().numpy() 

    init = data.reshape((-1,)).cpu().numpy()
    solution = integrate.solve_ivp(ode_solver_func_forward, (t_start, T), init, rtol=rtol, atol=atol, method=method)
    xT = solution.y[:, -1]
    score_T = prior_score(xT)
    init = np.concatenate([xT.reshape((-1,)), score_T.reshape((-1,))], axis=0).reshape((-1,))
    solution = integrate.solve_ivp(ode_solver_func_reverse, (T, t_start), init, rtol=rtol, atol=atol, method=method)
    nfe = solution.nfev
    t = solution.t
    zp = solution.y[:, -1]
    score_0 = zp[-shape[0]*shape[1]:].reshape((shape[0], shape[1]))
    return torch.from_numpy(score_0)


if not os.path.exists('toy_fig'):
    os.mkdir('toy_fig')

########################

# fig, axes = plt.subplots(nrows=3, figsize=(5, 14), constrained_layout=True)
# xt = draw_forward(axes[0], fig, 'ode', forward_score)
# draw_reverse(axes[1], fig, 'ode', forward_score, title="Gaussian")
# draw_reverse(axes[2], fig, 'ode', forward_score, start=torch.from_numpy(xt).to(device), title="qT")
# plt.savefig('toy_fig/ode.jpg')

########################

# fig, axes = plt.subplots(nrows=3, figsize=(5, 14), constrained_layout=True)
# xt = draw_forward(axes[0], fig, 'sde', forward_score)
# draw_reverse(axes[1], fig, 'sde', forward_score, title="Gaussian")
# draw_reverse(axes[2], fig, 'sde', forward_score, start=torch.from_numpy(xt).to(device), title="qT")
# plt.savefig('toy_fig/sde.jpg')

########################

# fig, axes = plt.subplots(nrows=2, figsize=(5, 10), constrained_layout=True)
# plt_flow_density(forward_log_q, axes[0], npts=1000, title="Forward log q(x)")
# # plt_flow_density(ode_likelihood, axes[1], npts=1000, title="ODE log q(x)")
# plt.savefig('toy_fig/q.jpg')

########################

# fig, axes = plt.subplots(nrows=2, figsize=(5, 10), constrained_layout=True)
# plt_score(forward_score, axes[0], npts=1000, title="Forward score", predict_eps=False)
# plt_score(forward_second_score, axes[1], npts=1000, title="Forward second score", predict_eps=False)
# plt.savefig('toy_fig/q_score.jpg')

# fig, axes = plt.subplots(nrows=2, figsize=(5, 10), constrained_layout=True)
# plt_score(forward_score, axes[0], npts=1000, title="Forward score", predict_eps=True)
# plt_score(forward_second_score, axes[1], npts=1000, title="Forward second score", predict_eps=True)
# plt.savefig('toy_fig/q_score_eps.jpg')

# fig, axes = plt.subplots(nrows=2, figsize=(5, 10), constrained_layout=True)
# plt_score(forward_score, axes[0], npts=1000, title="Forward score", predict_eps=True, eps_square=True)
# plt_score(forward_second_score, axes[1], npts=1000, title="Forward second score", predict_eps=True, eps_square=True)
# plt.savefig('toy_fig/q_score_eps_square.jpg')

########################

# fig, axes = plt.subplots()
# plt_marginal_density(forward_score, forward_second_score, axes, title="q(x) by ode")
# plt.savefig('toy_fig/marginal_density.jpg')

########################

# fig, axes = plt.subplots(nrows=2, figsize=(5, 10), constrained_layout=True)
# plt_importance_proposal_density(axes)
# # plt_score_trajectory(axes[0])
# plt.savefig('toy_fig/importance_distribution_from_0.01.jpg')


########################

def get_timestep_embedding(timesteps, embedding_dim=128):
    """
      From Fairseq.
      Build sinusoidal embeddings.
      This matches the implementation in tensor2tensor, but differs slightly
      from the description in Section 3.5 of "Attention Is All You Need".
      https://github.com/pytorch/fairseq/blob/master/fairseq/modules/sinusoidal_positional_embedding.py
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float, device=timesteps.device) * -emb)

    emb = timesteps.float().view(-1, 1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, [0,1])

    return emb


# class MLPResidualBlock(nn.Module):
#     def __init__(self, in_features: int, out_features: int, hidden_units: int, af=nn.ELU()):
#         super().__init__()
#         self.af = af
#         self.linear1 = nn.Linear(in_features, hidden_units)
#         self.linear2 = nn.Linear(hidden_units, out_features)
#         self.short_cut = nn.Linear(in_features, out_features)

#     def forward(self, inputs):
#         outputs = self.af(self.linear1(inputs))
#         outputs = self.af(self.linear2(outputs))
#         return outputs + self.short_cut(inputs)

# class MLPResidualNet(nn.Module):
#     def __init__(self, n_features_lst, af=nn.ELU()):
#         super().__init__()
#         modules = []
#         for i in range(len(n_features_lst) - 1):
#             modules.append(MLPResidualBlock(n_features_lst[i], n_features_lst[i + 1], n_features_lst[i], af))
#             if i < len(n_features_lst) - 2:
#                 modules.append(af)
#         self.net = nn.Sequential(*modules)

#     def forward(self, inputs):
#         return self.net(inputs)

class MLP(torch.nn.Module):
    def __init__(self, input_dim, layer_widths, activate_final = False, activation_fn=F.relu):
        super(MLP, self).__init__()
        layers = []
        prev_width = input_dim
        for layer_width in layer_widths:
            layers.append(torch.nn.Linear(prev_width, layer_width))
            # # same init for everyone
            # torch.nn.init.constant_(layers[-1].weight, 0)
            prev_width = layer_width
        self.input_dim = input_dim
        self.layer_widths = layer_widths
        self.layers = torch.nn.ModuleList(layers)
        self.activate_final = activate_final
        self.activation_fn = activation_fn
        
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation_fn(layer(x))
        x = self.layers[-1](x)
        if self.activate_final:
            x = self.activation_fn(x)
        return x

class ScoreNetwork(torch.nn.Module):

    def __init__(self, encoder_layers=[16], pos_dim=16, decoder_layers=[128,128], x_dim=1, act_fn=nn.SiLU):
        super().__init__()
        self.temb_dim = pos_dim
        t_enc_dim = pos_dim *2
        self.locals = [encoder_layers, pos_dim, decoder_layers, x_dim]

        self.net = MLP(2 * t_enc_dim,
                       layer_widths=decoder_layers +[x_dim],
                       activate_final = False,
                       activation_fn=act_fn())

        self.t_encoder = MLP(pos_dim,
                             layer_widths=encoder_layers +[t_enc_dim],
                             activate_final = False,
                             activation_fn=act_fn())

        self.x_encoder = MLP(x_dim,
                             layer_widths=encoder_layers +[t_enc_dim],
                             activate_final = False,
                             activation_fn=act_fn())

    def forward(self, x, t):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        temb = get_timestep_embedding(t, self.temb_dim)
        temb = self.t_encoder(temb)
        xemb = self.x_encoder(x)
        h = torch.cat([xemb ,temb], -1)
        out = self.net(h) 
        return out

def batch_data(train_data, batch_size=5000):
    indices = np.arange(train_data.shape[0])
    np.random.shuffle(indices)
    return train_data[indices[:batch_size]]


batch_size = 5000
log_val = 100
niters = 100000
viz_freq = 10000
# niters = 50000
# viz_freq = 2000

is_eval = False
is_resume = False
use_final = False

data_name = 'checkerboard'
LOW, HIGH = -4., 4.
x_dim = 2

# data_name = 'swissroll'
# LOW, HIGH = -7., 7.
# x_dim = 2

# data_name = 'mog'
# x_dim = 1

# train_data = inf_train_gen(data_name, dim=x_dim, batch_size=100).to(device)
# batch_size = 50

score_lambda = 0.5
third_score_lambda = 0.1

SDE_TYPE = 'VE'
# SDE_TYPE = 'VP'

reweighted = True

method = 'score_eps'
# method = 'score_eps_second'
# method = 'score_eps_second_jac'
# method = 'score_eps_third'

additional = '_from{}_{}_reweighted_{}_sigma_{}_{}'.format(START_TIME, SDE_TYPE, reweighted, sigma_0, sigma_1)


if not os.path.exists('experiments'):
    os.mkdir('experiments')

if not is_eval and not is_resume:
    resume = None
else:
    ckpt = 'checkpt_final.pth' if use_final else 'checkpt.pth'
    if 'second' in method:
        resume = 'experiments/{}_{}_{}_dim{}{}/{}'.format(data_name, method, score_lambda, x_dim, additional, ckpt)
    elif 'third' in method:
        resume = 'experiments/{}_{}_{}_{}_dim{}{}/{}'.format(data_name, method, score_lambda, third_score_lambda, x_dim, additional, ckpt)
    else:
        resume = 'experiments/{}_{}_dim{}{}/{}'.format(data_name, method, x_dim, additional, ckpt)


if 'second' in method:
    save_path = 'experiments/{}_{}_{}_dim{}{}'.format(data_name, method, score_lambda, x_dim, additional)
elif 'third' in method:
    save_path = 'experiments/{}_{}_{}_{}_dim{}{}'.format(data_name, method, score_lambda, third_score_lambda, x_dim, additional)
else:
    save_path = 'experiments/{}_{}_dim{}{}'.format(data_name, method, x_dim, additional)

# additional += '_usepretrain'
# resume = 'experiments/{}_{}_{}_{}_dim{}{}/checkpt.pth'.format(data_name, method, score_lambda, third_score_lambda, x_dim, additional)
# save_path += '_usepretrain'

print(save_path)

def eval_likelihood(model, times=1):
    score_fn, score_div_fn = get_score_fn_by_model(model)
    nll = []
    for _ in range(times):
        test_data = inf_train_gen(data=data_name, dim=x_dim, batch_size=5000).to(device)
        logpx = ode_likelihood(score_fn, score_div_fn, test_data, START_TIME)
        nll.append(logpx.mean().item())
    return -np.mean(nll)

def eval_scorenet(model, t, times=1):
    score_fn, score_div_fn = get_score_fn_by_model(model)
    score_jac_fn, score_grad_div_fn = get_score_grads_by_model(model)
    s_ode_list = []
    score_ode_with_normal_list = []
    s_with_normal_list = []
    s_div_with_normal_list = []
    s_grad_div_with_normal_list = []
    for _ in tqdm(range(times)):
        x = inf_train_gen(data=data_name, dim=x_dim, batch_size=5000).to(device)
        t_array = torch.ones((x.shape[0], 1)).to(x) * t
        eps = torch.randn_like(x).to(x)
        alpha_t = marginal_std_fn(t_array).reshape((-1, 1))
        x0_perm = marginal_mean_fn(x, t_array)
        xt = x0_perm + eps * alpha_t
        gt_square = g_square_fn(t_array).cpu()
        score_ode = ode_score(score_fn, score_jac_fn, score_grad_div_fn, xt, t).detach().cpu()
        score = score_fn(xt, t_array).detach().cpu()
        score_div = score_div_fn(xt, t_array).detach().cpu()
        score_grad_div = score_grad_div_fn(xt, t_array).detach().cpu()
        score_normal = (-xt / alpha_t / alpha_t).detach().cpu()
        score_normal_div = (-1. / alpha_t / alpha_t).detach().cpu()
        score_normal_grad_div = 0.

        s_ode = (norm_2(score_ode - score) * gt_square).mean().item()
        s_ode_list.append(s_ode)
        score_ode_with_normal = (gt_square * norm_2(score_ode - score_normal)).mean().item()
        score_ode_with_normal_list.append(score_ode_with_normal)
        s_with_normal = (gt_square * norm_2(score - score_normal)).mean().item()
        s_with_normal_list.append(s_with_normal)
        s_div_with_normal = (gt_square * norm_2(score_div - score_normal_div)).mean().item()
        s_div_with_normal_list.append(s_div_with_normal)
        s_grad_div_with_normal = (gt_square * norm_2(score_grad_div - score_normal_grad_div)).mean().item()
        s_grad_div_with_normal_list.append(s_grad_div_with_normal)
    return np.mean(s_ode_list), np.mean(score_ode_with_normal_list), np.mean(s_with_normal_list), np.mean(s_div_with_normal_list), np.mean(s_grad_div_with_normal)

def eval_fisher(model, t, times=1):
    score_fn, score_div_fn = get_score_fn_by_model(model)
    score_jac_fn, score_grad_div_fn = get_score_grads_by_model(model)
    fisher_list = []
    s_ode_list = []
    s_q_list = []
    inner_prod_list = []
    trace_differ_list = []
    first_differ_list = []
    ode_norm_list = []
    score_ode_with_normal_list = []
    score_q_with_normal_list = []
    s_with_normal_list = []
    s_div_with_normal_list = []
    s_grad_div_with_normal_list = []
    for _ in tqdm(range(times)):
        x = inf_train_gen(data=data_name, dim=x_dim, batch_size=5000)
        t_array = torch.ones_like(x).to(x) * t
        eps = torch.randn_like(x).to(x)
        alpha_t = marginal_std_fn(t_array).reshape((-1, 1))
        x0_perm = marginal_mean_fn(x, t_array)
        xt = x0_perm + eps * alpha_t
        gt_square = g_square_fn(t_array).cpu()
        score_ode = ode_score(score_fn, score_jac_fn, score_grad_div_fn, xt, t).detach().cpu()
        score_q = forward_score(xt, t_array).detach().cpu()
        score = score_fn(xt, t_array).detach().cpu()
        score_div = score_div_fn(xt, t_array).detach().cpu()
        score_grad_div = score_grad_div_fn(xt, t_array).detach().cpu()
        score_normal = (-xt / alpha_t / alpha_t).detach().cpu()
        score_normal_div = (-1. / alpha_t / alpha_t).detach().cpu()
        score_normal_grad_div = 0.

        fisher = (norm_2(score_ode - score_q) * gt_square).mean().item()
        fisher_list.append(fisher)
        s_ode = (norm_2(score_ode - score) * gt_square).mean().item()
        s_ode_list.append(s_ode)
        s_q = (norm_2(score - score_q) * gt_square).mean().item()
        s_q_list.append(s_q)
        kl_inner_prod = (gt_square * inner_prod(score - score_q, score_ode - score_q)).mean().item()
        inner_prod_list.append(kl_inner_prod)
        trace_differ = (gt_square * norm_2(score_div - forward_second_score(xt, t_array).detach().cpu())).mean().item()
        trace_differ_list.append(trace_differ)
        first_differ = (gt_square * inner_prod(score - score_q, score_ode)).mean().item()
        first_differ_list.append(first_differ)
        ode_norm = (gt_square * norm_2(score_ode)).mean().item()
        ode_norm_list.append(ode_norm)
        score_ode_with_normal = (gt_square * norm_2(score_ode - score_normal)).mean().item()
        score_ode_with_normal_list.append(score_ode_with_normal)
        score_q_with_normal = (gt_square * norm_2(score_q - score_normal)).mean().item()
        score_q_with_normal_list.append(score_q_with_normal)
        s_with_normal = (gt_square * norm_2(score - score_normal)).mean().item()
        s_with_normal_list.append(s_with_normal)
        s_div_with_normal = (gt_square * norm_2(score_div - score_normal_div)).mean().item()
        s_div_with_normal_list.append(s_div_with_normal)
        s_grad_div_with_normal = (gt_square * norm_2(score_grad_div - score_normal_grad_div)).mean().item()
        s_grad_div_with_normal_list.append(s_grad_div_with_normal)

    return np.mean(fisher_list), np.mean(s_ode_list), np.mean(s_q_list), np.mean(inner_prod_list), np.mean(trace_differ_list), np.mean(first_differ_list), np.mean(ode_norm_list), np.mean(score_ode_with_normal_list), np.mean(score_q_with_normal_list), np.mean(s_with_normal_list), np.mean(s_div_with_normal_list), np.mean(s_grad_div_with_normal)

def compute_data_entropy(times=5):
    kl_list = []
    fisher_list = []
    for _ in range(times):
        x = inf_train_gen(data=data_name, dim=x_dim, batch_size=5000)
        t = torch.ones_like(x).to(x) * T
        eps = torch.randn_like(x).to(x)
        alpha_t = marginal_std_fn(t).reshape((-1, 1))
        x0_perm = marginal_mean_fn(x, t)
        xt = x0_perm + eps * alpha_t
        kl = forward_log_q(xt, t) - prior_logp(xt)
        kl_list.append(kl.mean().item())
        fisher = (forward_score(xt, t) - prior_score(xt)).pow(2).sum(axis=1)
        fisher_list.append(fisher.mean().item())
    print("KL(q_T||p_T):", np.mean(kl_list))
    print("F(q_T||p_T):", np.mean(fisher_list))

    nll = []
    for _ in range(times):
        test_data = inf_train_gen(data=data_name, dim=x_dim, batch_size=5000)
        logpx = forward_log_q(test_data, torch.ones_like(test_data) * START_TIME)
        nll.append(-logpx.mean().item())
    print("NLL for q0:", np.mean(nll))

def likelihood_importance_cum_weight(t, eps):
    exponent1 = 0.5 * eps * (eps - 2) * beta_0 - 0.5 * eps ** 2 * beta_1
    exponent2 = 0.5 * t * (t - 2) * beta_0 - 0.5 * t ** 2 * beta_1
    term1 = np.where(np.abs(exponent1) <= 1e-3, -exponent1, 1. - np.exp(exponent1))
    term2 = np.where(np.abs(exponent2) <= 1e-3, -exponent2, 1. - np.exp(exponent2))
    return 0.5 * (-2 * np.log(term1) + 2 * np.log(term2)
                    + beta_0 * (-2 * eps + eps ** 2 - (t - 2) * t)
                    + beta_1 * (-eps ** 2 + t ** 2))

def sample_importance_weighted_time_for_likelihood(size, eps=START_TIME, steps=100):
    Z = likelihood_importance_cum_weight(T, eps)
    quantile = np.random.uniform(0., Z, size)
    lb = np.ones_like(quantile) * eps
    ub = np.ones_like(quantile) * T

    def bisection_func(carry, idx):
        lb, ub = carry
        mid = (lb + ub) / 2.
        value = likelihood_importance_cum_weight(mid, eps=eps)
        lb = np.where(value <= quantile, mid, lb)
        ub = np.where(value <= quantile, ub, mid)
        return (lb, ub), idx

    def scan(f, init, xs, length=None):
        if xs is None:
            xs = [None] * length
        carry = init
        ys = []
        for x in xs:
            carry, y = f(carry, x)
            ys.append(y)
        return carry, np.stack(ys)

    (lb, ub), _ = scan(bisection_func, (lb, ub), np.arange(0, steps))
    return (lb + ub) / 2.

def get_model_div(model, x, t):
    x.requires_grad_(True)
    model_output = model(x, t)
    if x_dim == 1:
        model_div = torch.autograd.grad(torch.sum(model_output), x, create_graph=True)[0]
    elif x_dim == 2:
        model_jac = batch_jacobian(model_output, x)
        model_div = batch_trace(model_jac)
    else:
        v = torch.randint(0, 2, x.shape).to(x) * 2 - 1
        Jv = torch.autograd.grad(torch.sum(model_output * v), x, create_graph=True)[0]
        model_div = torch.sum((v * Jv).reshape(x.shape[0], -1), -1, keepdim=True)
    x.requires_grad_(False)
    return model_div.reshape((-1, 1))

def get_model_jac(model, x, t):
    x.requires_grad_(True)
    model_output = model(x, t)
    if x_dim == 1:
        model_jac = torch.autograd.grad(torch.sum(model_output), x, create_graph=True)[0]
    elif x_dim == 2:
        model_jac = batch_jacobian(model_output, x)
    x.requires_grad_(False)
    return model_jac.reshape((x.shape[0], x_dim, x_dim))

def get_model_grad_div(model, x, t):
    x.requires_grad_(True)
    model_output = model(x, t)
    if x_dim == 1:
        model_div = torch.autograd.grad(torch.sum(model_output), x, create_graph=True)[0]
    elif x_dim == 2:
        model_jac = batch_jacobian(model_output, x)
        model_div = batch_trace(model_jac)
    model_div = model_div.reshape((-1, 1))
    model_div_grad = torch.autograd.grad(torch.sum(model_div), x, create_graph=True)[0]
    x.requires_grad_(False)
    return model_div_grad.reshape((x.shape[0], x_dim))

def get_score_fn_by_model(model):
    if method == 'score':
        def second_score(x, t):
            score_div = get_model_div(model, x, t)
            return score_div
        return model, second_score
    elif method == 'score_eps' or method == 'score_eps_second' or method == 'score_eps_third' or method == 'score_eps_second_jac':
        def score(x, t):
            alpha_t = marginal_std_fn(t).reshape((-1, 1))
            return model(x, t) / alpha_t
        def second_score(x, t):
            model_div = get_model_div(model, x, t)
            alpha_t = marginal_std_fn(t).reshape((-1, 1))
            return model_div / alpha_t
        return score, second_score

def get_score_grads_by_model(model):
    if method == 'score':
        def score_jac_fn(x, t):
            score_jac = get_model_jac(model, x, t)
            return score_jac
        def score_div_grad_fn(x, t):
            score_div_grad = get_model_grad_div(model, x, t)
            return score_div_grad
        return score_jac_fn, score_div_grad_fn
    elif method == 'score_eps' or method == 'score_eps_second' or method == 'score_eps_third' or method == 'score_eps_second_jac':
        def score_jac_fn(x, t):
            alpha_t = marginal_std_fn(t).reshape((-1, 1, 1))
            model_jac = get_model_jac(model, x, t)
            return model_jac / alpha_t
        def score_div_grad_fn(x, t):
            alpha_t = marginal_std_fn(t).reshape((-1, 1))
            model_div_grad = get_model_grad_div(model, x, t)
            return model_div_grad / alpha_t
        return score_jac_fn, score_div_grad_fn

def batch_jacobian(g, x, create_graph=True):
    jac = []
    for d in range(g.shape[1]):
        jac.append(torch.autograd.grad(torch.sum(g[:, d]), x, create_graph=create_graph)[0].view(x.shape[0], 1, x.shape[1]))
    return torch.cat(jac, 1)

def batch_trace(M):
    return M.view(M.shape[0], -1)[:, ::M.shape[1] + 1].sum(1)

def norm_2(x):
    return torch.sum(torch.square(x).reshape((x.shape[0], -1)), -1, keepdim=True)

def inner_prod(x, y):
    return torch.sum((x * y).reshape((x.shape[0], -1)), -1, keepdim=True) 

def loss_fn(model, itr):
    x = inf_train_gen(data=data_name, dim=x_dim, batch_size=batch_size).to(device)
    # x = batch_data(train_data, batch_size)
    eps = torch.randn_like(x).to(device)
    if SDE_TYPE == 'VE' or reweighted:
        t = np.random.uniform(START_TIME, T, x.shape[0])
    else:
        t = sample_importance_weighted_time_for_likelihood(x.shape[0], eps=START_TIME)
    t = torch.from_numpy(t).to(device).float()
    alpha_t = marginal_std_fn(t).reshape((-1, 1))
    x0_perm = marginal_mean_fn(x, t)
    xt = x0_perm + eps * alpha_t
    xt_neg = x0_perm - eps * alpha_t
    score_fn, second_score_fn = get_score_fn_by_model(model)

    if method == 'score':
        score = score_fn(xt, t)
        loss_1 = torch.square(score * alpha_t + eps)
        loss_1 = torch.mean(loss_1.reshape((loss_1.shape[0], -1)), axis=-1)
        loss = loss_1
    elif method == 'score_eps':
        model_output = model(xt, t)
        loss_1 = torch.square(model_output + eps)
        loss_1 = torch.mean(loss_1.reshape((loss_1.shape[0], -1)), axis=-1)
        loss = loss_1
        loss_2 = torch.zeros_like(loss_1)
        loss_3 = loss_2
    elif method == 'score_eps_second':
        model_output = model(xt, t)
        loss_1 = torch.square(model_output + eps)
        loss_1 = torch.mean(loss_1.reshape((loss_1.shape[0], -1)), axis=-1)
        model_div = get_model_div(model, xt, t).reshape((xt.shape[0], 1))
        model_output_fixed = model_output.detach()
        model_output_2 = model_output_fixed.pow(2)
        model_output_2 = torch.sum(model_output_2.reshape((model_output_2.shape[0], -1)), -1, keepdim=True)
        eps_2 = torch.sum(eps.pow(2).reshape((eps.shape[0], -1)), -1, keepdim=True)
        model_eps_product = torch.sum((model_output_fixed * eps).reshape(x.shape[0], -1), -1, keepdim=True)
        loss_2 = torch.square(alpha_t * model_div + x_dim - eps_2 - 2 * model_eps_product - model_output_2) / x_dim
        loss_2 = loss_2.reshape((loss_2.shape[0],))
        loss = loss_1 + score_lambda * loss_2
        loss_3 = torch.zeros_like(loss_1)
    elif method == 'score_eps_second_jac':
        model_output = model(xt, t)
        loss_1 = torch.square(model_output + eps)
        loss_1 = torch.mean(loss_1.reshape((loss_1.shape[0], -1)), axis=-1)
        model_jac = get_model_jac(model, xt, t)
        model_output = model_output.detach()
        I = torch.eye(x_dim).reshape((1, x_dim, x_dim)).repeat(xt.shape[0], 1, 1).reshape((xt.shape[0], int(x_dim * x_dim))).to(xt).float()
        cond_score = eps + model_output
        cond_score_matrix = torch.bmm(cond_score.reshape((xt.shape[0], x_dim, 1)), cond_score.reshape((xt.shape[0], 1, x_dim))).reshape((xt.shape[0], int(x_dim * x_dim)))
        loss_2 = torch.mean(torch.square(alpha_t * model_jac.reshape((xt.shape[0], int(x_dim * x_dim))) + I - cond_score_matrix), axis=-1)
        loss_2 = loss_2.reshape((loss_2.shape[0],))
        loss = loss_1 + score_lambda * loss_2
        loss_3 = torch.zeros_like(loss_1)
    elif method == 'score_eps_third':
        model_output = model(xt, t)
        loss_1 = torch.square(model_output + eps)
        loss_1 = torch.mean(loss_1.reshape((loss_1.shape[0], -1)), axis=-1)

        model_div = get_model_div(model, xt, t).reshape((xt.shape[0], 1))
        model_jac = get_model_jac(model, xt, t)
        model_output = model_output.detach()
        I = torch.eye(x_dim).reshape((1, x_dim, x_dim)).repeat(xt.shape[0], 1, 1).reshape((xt.shape[0], int(x_dim * x_dim))).to(xt).float()
        cond_score = eps + model_output
        cond_score_matrix = torch.bmm(cond_score.reshape((xt.shape[0], x_dim, 1)), cond_score.reshape((xt.shape[0], 1, x_dim))).reshape((xt.shape[0], int(x_dim * x_dim)))
        loss_2 = torch.mean(torch.square(alpha_t * model_jac.reshape((xt.shape[0], int(x_dim * x_dim))) + I - cond_score_matrix), axis=-1)
        loss_2 = loss_2.reshape((loss_2.shape[0],))

        alpha_t_square = alpha_t.pow(2)
        model_grad_div = get_model_grad_div(model, xt, t)
        model_output = model_output.detach()
        model_div = model_div.detach()
        cond_score = eps + model_output
        model_jac = model_jac.detach()
        I = torch.eye(x_dim).reshape((1, x_dim, x_dim)).repeat((xt.shape[0], 1, 1)).to(xt).float()
        loss_3 = alpha_t_square * model_grad_div + norm_2(cond_score) * cond_score - (alpha_t * model_div + x_dim) * cond_score - 2 * torch.bmm(alpha_t.reshape((xt.shape[0], 1, 1)) * model_jac + I, cond_score.reshape((xt.shape[0], x_dim, 1))).reshape((xt.shape[0], x_dim))
        loss_3 = torch.mean(torch.square(loss_3).reshape((xt.shape[0], -1)), 1)
        loss = loss_1 + score_lambda * loss_2 + third_score_lambda * loss_3

    return torch.mean(loss), torch.mean(loss_1), torch.mean(score_lambda * loss_2), torch.mean(third_score_lambda * loss_3)


def visualize(score_fn, second_score_fn, itr, ema):
    if x_dim == 1:
        with torch.no_grad():
            fig, axes = plt.subplots(nrows=3, figsize=(5, 14), constrained_layout=True)
            xt = draw_forward(axes[0], fig, 'ode', forward_score)
            draw_reverse(axes[1], fig, 'ode', score_fn, title="Gaussian")
            draw_reverse(axes[2], fig, 'ode', score_fn, start=torch.from_numpy(xt).to(device), title="qT")
            plt.savefig(os.path.join(save_path, '{}_ode_trajectory.jpg'.format(itr)))

            fig, axes = plt.subplots(nrows=3, figsize=(5, 14), constrained_layout=True)
            xt = draw_forward(axes[0], fig, 'sde', forward_score)
            draw_reverse(axes[1], fig, 'sde', score_fn, title="Gaussian")
            draw_reverse(axes[2], fig, 'sde', score_fn, start=torch.from_numpy(xt).to(device), title="qT")
            plt.savefig(os.path.join(save_path, '{}_sde_trajectory.jpg'.format(itr)))
        
        
        fig, axes = plt.subplots(nrows=2, figsize=(5, 10), constrained_layout=True)
        plt_marginal_density(forward_score, forward_second_score, axes[0], title="q(x) by ode", LOW=LOW, HIGH=HIGH)
        plt_marginal_density(score_fn, second_score_fn, axes[1], title="p(x) by ode", LOW=LOW, HIGH=HIGH)
        plt.savefig(os.path.join(save_path, '{}_marginal.jpg'.format(itr)))

        # fig, axes = plt.subplots(nrows=2, figsize=(5, 10), constrained_layout=True)
        # with torch.no_grad():
        #     plt_score(score_fn, axes[0], npts=1000, title="Forward score", predict_eps=False)
        # plt_score(second_score_fn, axes[1], npts=1000, title="Forward second score", predict_eps=False)
        # plt.savefig(os.path.join(save_path, '{}_score.jpg'.format(itr)))

        # fig, axes = plt.subplots(nrows=2, figsize=(5, 10), constrained_layout=True)
        # with torch.no_grad():
        #     plt_score(score_fn, axes[0], npts=1000, title="Forward score", predict_eps=True)
        # plt_score(second_score_fn, axes[1], npts=1000, title="Forward second score", predict_eps=True)
        # plt.savefig(os.path.join(save_path, '{}_score_eps.jpg'.format(itr)))

        # fig, axes = plt.subplots(nrows=2, figsize=(5, 10), constrained_layout=True)
        # with torch.no_grad():
        #     plt_score(score_fn, axes[0], npts=1000, title="Forward score", predict_eps=True, eps_square=True)
        # plt_score(second_score_fn, axes[1], npts=1000, title="Forward second score", predict_eps=True, eps_square=True)
        # plt.savefig(os.path.join(save_path, '{}_score_eps_square.jpg'.format(itr)))
    elif x_dim == 2:
        fig, axes = plt.subplots(nrows=2, figsize=(5, 10), constrained_layout=True)
        px = plt_marginal_density(score_fn, second_score_fn, axes[0], title="p(x) by ode", npts=400, memory=20000, dim=x_dim, LOW=LOW, HIGH=HIGH)
        # plt_marginal_samples(score_fn, axes[0], title="samples by ode")
        plt.savefig(os.path.join(save_path, '{}_marginal.jpg'.format(itr)))
        np.save(os.path.join(save_path, '{}_density'.format(itr)), px)


def train(model, optimizer, ema):
    model.train()
    best_nll = 1e10
    for itr in range(1, niters + 1):
        loss, loss1, loss2, loss3 = loss_fn(model, itr)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        optimizer.zero_grad()
        if itr > 500:
            ema.apply()
        if itr % log_val == 0:
            print('itr: %d, loss: %.5e, loss1: %.5e, loss2: %.5e, loss3: %.5e gnorm: %.5e' % (itr, loss.item(), loss1.item(), loss2.item(), loss3.item(), grad_norm))

        # if itr == 1 or itr % viz_freq == 0:
        if itr % viz_freq == 0:
            model.eval()
            ema.swap()
            score_fn, second_score_fn = get_score_fn_by_model(model)
            visualize(score_fn, second_score_fn, itr, ema)
            nll = eval_likelihood(model)
            print("itr: %d, nll: %.5e, best_nll: %.5e" % (itr, nll, best_nll))
            ema.swap()
            if itr > 1 and best_nll > nll:
                print('saving ckpt...')
                torch.save({
                            'state_dict': model.state_dict(),
                            'ema': ema,
                        }, os.path.join(save_path, 'checkpt.pth'))
                best_nll = nll
            torch.save({
                        'state_dict': model.state_dict(),
                        'ema': ema,
                    }, os.path.join(save_path, 'checkpt_final.pth'))
            model.train()


def evaluate(model, ema):
    model.eval()
    ema.swap()
    if x_dim == 1:
        fig, axes = plt.subplots(nrows=2, figsize=(5, 10), constrained_layout=True)
        score_fn, _ = get_score_fn_by_model(model)
        score_jac_fn, score_div_grad = get_score_grads_by_model(model)
        plt_marginal_score(None, None, None, axes[0], title="data score", LOW=LOW, HIGH=HIGH, is_data=True)
        plt_marginal_score(score_fn, score_jac_fn, score_div_grad, axes[1], title="score by ode", LOW=LOW, HIGH=HIGH)
        plt.savefig(os.path.join(save_path, '{}_score.jpg'.format(0)))
    for t in [0.5, 0.6, 0.7, 0.8, 0.9, 1.]:
        if x_dim == 1:
            fisher, s_ode, s_q, kl_inner_prod, second_differ, first_differ, ode_norm, score_ode_normal, score_q_normal, s_normal, s_div_normal, s_grad_div_normal = eval_fisher(model, t, times=5)
            print("t=%.5e, fisher: %.5e, s-ode: %.5e, s-q: %.5e, kl-inner: %.5e, second_differ: %.5e, first_differ: %.5e, ode_norm: %.5e, score_ode_normal: %.5e, score_q_normal: %.5e, s_normal: %.5e, s_div_normal: %.5e, s_grad_div_normal: %.5e" % (t, fisher, s_ode, s_q, kl_inner_prod, second_differ, first_differ, ode_norm, score_ode_normal, score_q_normal, s_normal, s_div_normal, s_grad_div_normal))
        else:
            s_ode, score_ode_normal, s_normal, s_div_normal, s_grad_div_normal = eval_scorenet(model, t, times=5)
            print("t=%.5e, s-ode: %.5e, score_ode_normal: %.5e, s_normal: %.5e, s_div_normal: %.5e, s_grad_div_normal: %.5e" % (t, s_ode, score_ode_normal, s_normal, s_div_normal, s_grad_div_normal))
    score_fn, second_score_fn = get_score_fn_by_model(model)
    nll = eval_likelihood(model, times=5)
    print("nll: %.5e" % (nll))
    visualize(score_fn, second_score_fn, 0, ema)
    ema.swap()


def main():
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    model = ScoreNetwork(encoder_layers=[128, 128], pos_dim=64, decoder_layers=[512, 512], x_dim=x_dim, act_fn=nn.SiLU).to(device)
    print(model)
    ema = utils_toy.ExponentialMovingAverage(model)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    if resume is not None:
        print('resume model', resume)
        checkpt = torch.load(resume)
        state = model.state_dict()
        state.update(checkpt['state_dict'])
        model.load_state_dict(state, strict=True)
        ema.set(checkpt['ema'])
        del checkpt
        del state

    if is_eval:
        evaluate(model, ema)
    else:
        train(model, optimizer, ema)

main()

