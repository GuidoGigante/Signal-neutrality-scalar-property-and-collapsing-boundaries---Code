import numpy as np
import pandas as pd
from numba import jit
import numba.typed as numba_type
from scipy import special
import time

# Simple object for function timing
class TicToc(object):
    def __init__(self):
        self.last_t = time.time()

    def tic(self):
        self.last_t = time.time()
        return

    def toc(self, b_print=True):
        elapsed = time.time() - self.last_t
        if b_print:
            print(f"Elapsed {elapsed:.1f} seconds.")
        return elapsed


import math


# Generic function to make lambda (in TD(lambda)) increase as a power-law
# from a minimum value lambda_min to a maximum value lambda_max
@jit(nopython=True)
def _lambda_f(
    iter_no,
    lambda_min,
    lambda_max,
    iter_scale=1 * 10 ** 5 - 3 * 10 ** 4,
    delta_step=10 ** 4,
    iter_gamma=0.5,
):
    iter_no_x = int(iter_no / delta_step) * delta_step

    # Lambda increases from lambda_min to lambda_max as a power (iter_gamma) with a "time-scale" (iter_scale)
    lam = lambda_min + (iter_no_x / iter_scale) ** iter_gamma * (
        lambda_max - lambda_min
    )

    lam = min(lam, lambda_max)
    return lam

## The following function defines how the learning rate changes across different episodes. 
@jit(nopython=True)
def _alpha_f(
    iter_no, alpha0, iter_scale=10 ** 4, iter_gamma=0.75, lam=None, lam0=None
):
    alpha = alpha0 / (1 + iter_no / iter_scale) ** iter_gamma

    if lam is not None:
        alpha *= (1 - lam) / (1 - lam0)

    return alpha

# Update of the signal integrators by one time step
@jit(nopython=True)
def _update_integrators(ints, ints_n, signal, t, i_taus, sig_i):
    n_taus = ints.shape[1]

    if t > 0:
        ints[t, :] = ints[t - 1, :] - (ints[t - 1, :] - signal) * i_taus
    else:
        ints[t, :] = signal * i_taus

    ints_n[t, :] = ints[t, :] + sig_i * np.random.randn(n_taus)

# Update of the clock integrators by one time step
@jit(nopython=True)
def _update_t_integrators(ints_t, ints_t_n, t, sig_i):
    n_taus = ints_t.shape[1]
    ints_t_n[t, :] = ints_t[t, :] + sig_i * np.random.randn(n_taus)


# Computation of the probabilities for the different actions (policy; 0="right", 1="left", 2="weight").
# If you have n_taus different time-scales:
# w: signal integrators weights (3 x n_taus)
# wt: time integrators weights (3 x n_taus)
# w_b: biases (3)
# ints_n: instantaneous value of the signal integrators (corrupted by intrinsic noise - "_n")
# ints_t_n: instantaneous value of the time integrators (corrupted by intrinsic noise - "_n")
# t: time step during the episode
# use_t_integrators: if False, ints_t_n are not used
# p3_tot_abs: True if you want signal integrators for "wait" to enter as abs(w @ ints_n);
#   False if you want w @ abs(ints_n) (this is the choice in the paper)   
@jit(nopython=True)
def _compute_ps(
    w, w_t, w_b, ints_n, ints_t_n, t, use_t_integrators=True, p3_tot_abs=False
):
    n_taus = ints_n.shape[1]

    ix = ints_n[t, :]
    if use_t_integrators:
        itx = ints_t_n[t, :]

    phs_sign = np.ones((3,))
    if p3_tot_abs:
        ps = w @ ix
        phs_sign[2] = np.sign(ps[2])
        ps[2] = np.abs(ps[2])
    else:
        ps = np.zeros((3,))
        ps[:2] = w[:2, :] @ ix
        ps[2] = w[2, :] @ np.abs(ix)

    if use_t_integrators:
        ps += w_t @ itx
    ps += w_b
    ps -= ps.max()
    ps = np.exp(ps)
    ps[:] /= ps.sum()
    return ps, phs_sign


# Computation of the V value function
# th: signal integrators weights (3 x n_taus)
# th_t: time integrators weights (3 x n_taus)
# th_b: biases (3)
# Other parameters: see _compute_ps
@jit(nopython=True)
def _compute_v(
    th,
    th_t,
    th_b,
    ints_n,
    ints_t_n,
    t,
    use_t_integrators=True,
    p3_tot_abs=False,
):
    ix = ints_n[t, :]
    if use_t_integrators:
        itx = ints_t_n[t, :]

    if p3_tot_abs:
        v = th @ ix
        vh_sign = np.sign(v)
        v = np.abs(v)
    else:
        v = th @ np.abs(ix)
        vh_sign = 1

    v += th_b

    if use_t_integrators:
        v += th_t @ itx

    return v, vh_sign


# Utility function, returns True if value i is in array v
# (introduced because the equivalente numpy function is not
# supported in numba to the best of our knowledge)
@jit(nopython=True)
def _isin(i, v):
    for j in v:
        if j == i:
            return True

    return False

# Function to train with Adam optimizer (on K episodes).
# Given as input the parameters of an agent, the function uses Adam 
# optimizer to compute the update for all the different parameters.
# ws0: initial weights for signal integrators (3 x (2 x n_taus + 1)) for policy
#   format:
#       ws0[a, :n_taus]: signal integrators weights for action a (a = 0, 1, 2)
#       ws0[a, n_taus: 2 * n_taus]: time integrators weights for action a
#       ws0[a, 2 * n_taus]: bias for action a
#   WARNING: what here is called w (for weight) in the paper is named theta
# ths0: initial weights for signal integrators (3 x n_taus) for value function
#   (same format as ws0)
# taus: the time-scales for integrators (n_taus)
# dt: the time-step for simulation (in the paper, 10 * 10 ** -3 s)
# t_maxs: an array (K) with the max duration for each of the K episodes
#   (in the paper, all the t_maxs are identical to a value t_max - that may change
#    from case to case)
# sig: the standard deviation of the signal noise
# sig_i: the standard deviation of the intrinsic noise
# gamma: reward discount factor (in the paper always very close to 1)
# mus: array (K) with the value of the signal mean for each of the K episodes
#   (mus are typically extracted by a given distribution; in the paper,
#    a Gaussian of mean 0 and standard deviation sigma_mu)
# rewards: array (K) with the value of the reward for a correct decision
#   for each of the K episodes (in the paper, rewards = np.ones((K,)))
# use_t_integrators: if False, ints_t_n are not used
# p3_tot_abs: True if you want signal integrators for "wait" to enter as abs(w @ ints_n);
#   False if you want w @ abs(ints_n) (this is the choice in the paper)
# alpha0: initial learning rate
# iter_scale, iter_gamma: parameters that determine how the learning rate decays (see _alpha_f)
# lambda_min, lambda_max: min and max value of the lambda of TD(lambda)
#   (in the paper we used the default value, lambd_min=lambda_max≈1)
# beta_1, beta_2, eps_adam: as in original Adam paper (https://arxiv.org/pdf/1412.6980.pdf)
# w_history_episodes: array with an increasing set of values (in [0, K)) marking the
#   episodes during training in which you want a "snapshot" of the weights of the agent
#   (to have information on how the training has evolved)
# b_exp_i_taus: if True, Eqs. 6 and 7 in the paper are integrated "exactly"
#   (assuming the signal is constant during each time step) at each time-step;
#   if False, they are integrated with Euler method (in the paper, False)
#
# For output, see train_with_adam
@jit(nopython=True)
def _train_with_adam(
    ws0,
    ths0,
    taus,
    dt,
    t_maxs,
    sig,
    sig_i,
    gamma,
    mus,
    rewards,
    use_t_integrators=True,
    p3_tot_abs=False,
    alpha0=0.001,
    iter_scale=10 ** 4,
    iter_gamma=0.75,
    lambda_min=0.99999,
    lambda_max=1.0 - 1.0 / 100000,
    beta_1=0.9,
    beta_2=0.999,
    eps_adam=10 ** -8,
    w_history_episodes=None,
    b_exp_i_taus=False,
):

    n_taus = taus.size
    n_max_t_max = int(np.ceil(t_maxs.max() / dt))
    n_episodes = min(mus.size, rewards.size, t_maxs.size)

    if not b_exp_i_taus:
        i_taus = dt / taus
    else:
        i_taus = 1.0 - np.exp(-dt / taus)

    w = ws0[:, :n_taus].copy()
    w_t = ws0[:, n_taus : 2 * n_taus].copy()
    w_b = ws0[:, -1].copy()

    ws = np.zeros_like(ws0)
    ths = np.zeros_like(ths0)

    th = ths0[:n_taus].copy()
    th_t = ths0[n_taus : 2 * n_taus].copy()
    th_b = ths0[-1]

    ints = np.zeros((n_max_t_max, n_taus))
    ints_n = np.zeros((n_max_t_max, n_taus))
    ints_t = np.zeros((n_max_t_max, n_taus))
    ints_t_n = np.zeros((n_max_t_max, n_taus))

    ts = np.arange(0, n_max_t_max)
    for tau_no in range(n_taus):
        ints_t[:, tau_no] = 1.0 - np.exp(-ts * dt / taus[tau_no])

    def lambda_w(iter_no):
        return _lambda_f(iter_no, lambda_min=lambda_min, lambda_max=lambda_max)

    def lambda_th(iter_no):
        return _lambda_f(iter_no, lambda_min=lambda_min, lambda_max=lambda_max)

    lam_w0 = lambda_w(0)
    lam_th0 = lambda_th(0)

    def alpha_w(iter_no, lam):
        return _alpha_f(
            iter_no,
            alpha0=alpha0,
            iter_scale=iter_scale,
            iter_gamma=iter_gamma,
            lam=lam,
            lam0=lam_w0,
        )

    def alpha_th(iter_no, lam):
        return _alpha_f(
            iter_no,
            alpha0=alpha0,
            iter_scale=iter_scale,
            iter_gamma=iter_gamma,
            lam=lam,
            lam0=lam_th0,
        )

    z_w = np.zeros(w.shape)
    z_w_t = np.zeros(w_t.shape)
    z_w_b = np.zeros(w_b.shape)

    z_th = np.zeros(th.shape)
    z_th_t = np.zeros(th_t.shape)
    z_th_b = 0.0

    m_w = np.zeros_like(w)
    m_w_t = np.zeros_like(w_t)
    m_w_b = np.zeros_like(w_b)
    v_w = np.zeros_like(w)
    v_w_t = np.zeros_like(w_t)
    v_w_b = np.zeros_like(w_b)
    m_w_hat = np.zeros_like(w)
    m_w_t_hat = np.zeros_like(w_t)
    m_w_b_hat = np.zeros_like(w_b)
    v_w_hat = np.zeros_like(w)
    v_w_t_hat = np.zeros_like(w_t)
    v_w_b_hat = np.zeros_like(w_b)

    m_th = np.zeros_like(th)
    m_th_t = np.zeros_like(th_t)
    m_th_b = 0.0
    v_th = np.zeros_like(th)
    v_th_t = np.zeros_like(th_t)
    v_th_b = 0.0
    m_th_hat = np.zeros_like(th)
    m_th_t_hat = np.zeros_like(th_t)
    m_th_b_hat = 0.0
    v_th_hat = np.zeros_like(th)
    v_th_t_hat = np.zeros_like(th_t)
    v_th_b_hat = 0.0

    beta_1_n = beta_1
    beta_2_n = beta_2

    episodes = np.zeros((n_episodes, 4))

    history_episodes = numba_type.List()
    w_history = numba_type.List()
    th_history = numba_type.List()

    iter_no = 0
    if w_history_episodes is not None and _isin(iter_no, w_history_episodes):
        ws[:, :n_taus] = w
        ws[:, n_taus : 2 * n_taus] = w_t
        ws[:, -1] = w_b

        ths[:n_taus] = th
        ths[n_taus : 2 * n_taus] = th_t
        ths[-1] = th_b

        history_episodes.append(iter_no)
        w_history.append(ws.copy())
        th_history.append(ths.copy())

    for iter_no in range(n_episodes):
        t = 0
        mu = mus[iter_no]
        reward = rewards[iter_no]
        t_max = t_maxs[iter_no]
        n_t_max = int(np.ceil(t_max / dt))

        signal = mu + sig * np.random.randn()
        _update_integrators(ints, ints_n, signal, t, i_taus, sig_i)
        if use_t_integrators:
            _update_t_integrators(ints_t, ints_t_n, t, sig_i)

        gamma_n = 1.0
        z_w[:, :] = 0.0
        z_w_t[:, :] = 0.0
        z_w_b[:] = 0.0
        z_th[:] = 0.0
        z_th_t[:] = 0.0
        z_th_b = 0.0

        r = 0.0
        while t < n_t_max:
            ps, phs_sign = _compute_ps(
                w, w_t, w_b, ints_n, ints_t_n, t, use_t_integrators, p3_tot_abs
            )
            #             a = np.random.choice(3, size=1, replace=True, p=ps)[0]
            a = np.searchsorted(ps.cumsum(), np.random.rand())

            v0, v0h_sign = _compute_v(
                th,
                th_t,
                th_b,
                ints_n,
                ints_t_n,
                t,
                use_t_integrators,
                p3_tot_abs,
            )

            if a == 0 and mu > 0.0:
                r = reward
            elif a == 1 and mu < 0.0:
                r = reward

            dv0 = 1.0

            delta = r - v0

            if a == 0 or a == 1 or t == n_t_max - 1:
                v = 0.0
            else:
                signal = mu + sig * np.random.randn()
                _update_integrators(ints, ints_n, signal, t + 1, i_taus, sig_i)
                if use_t_integrators:
                    _update_t_integrators(ints_t, ints_t_n, t + 1, sig_i)
                v, _ = _compute_v(
                    th,
                    th_t,
                    th_b,
                    ints_n,
                    ints_t_n,
                    t + 1,
                    use_t_integrators,
                    p3_tot_abs,
                )
                delta += gamma * v

            lam_w = lambda_w(iter_no)
            a_w = alpha_w(iter_no, lam_w)
            lam_th = lambda_th(iter_no)
            a_th = alpha_th(iter_no, lam_th)

            for ax in range(3):
                dp = -ps[ax]
                if ax == a:
                    dp += 1.0

                if p3_tot_abs:
                    z_w[ax, :] = (
                        gamma * lam_w * z_w[ax, :]
                        + gamma_n * phs_sign[ax] * dp * ints_n[t, :]
                    )
                elif ax == 2:
                    z_w[ax, :] = gamma * lam_w * z_w[
                        ax, :
                    ] + gamma_n * dp * np.abs(ints_n[t, :])
                else:
                    z_w[ax, :] = (
                        gamma * lam_w * z_w[ax, :]
                        + gamma_n * dp * ints_n[t, :]
                    )

                if use_t_integrators:
                    z_w_t[ax, :] = (
                        gamma * lam_w * z_w_t[ax, :]
                        + gamma_n * dp * ints_t_n[t, :]
                    )

                z_w_b[ax] = gamma * lam_w * z_w_b[ax] + gamma_n * dp

            if p3_tot_abs:
                z_th[:] = (
                    gamma * lam_th * z_th[:] + dv0 * v0h_sign * ints_n[t, :]
                )
            else:
                z_th[:] = gamma * lam_th * z_th[:] + dv0 * np.abs(ints_n[t, :])

            if use_t_integrators:
                z_th_t[:] = gamma * lam_th * z_th_t[:] + dv0 * ints_t_n[t, :]
            z_th_b = gamma * lam_th * z_th_b + dv0

            m_w[:, :] = beta_1 * m_w + (1 - beta_1) * delta * z_w
            v_w[:, :] = beta_2 * v_w + (1 - beta_2) * np.power(delta * z_w, 2)
            m_w_hat[:, :] = m_w / (1 - beta_1_n)
            v_w_hat[:, :] = v_w / (1 - beta_2_n)
            w[:, :] += a_w * m_w_hat / (np.sqrt(v_w_hat) + eps_adam)

            m_w_t[:, :] = beta_1 * m_w_t + (1 - beta_1) * delta * z_w_t
            v_w_t[:, :] = beta_2 * v_w_t + (1 - beta_2) * np.power(
                delta * z_w_t, 2
            )
            m_w_t_hat[:, :] = m_w_t / (1 - beta_1_n)
            v_w_t_hat[:, :] = v_w_t / (1 - beta_2_n)
            w_t[:, :] += a_w * m_w_t_hat / (np.sqrt(v_w_t_hat) + eps_adam)

            m_w_b[:] = beta_1 * m_w_b + (1 - beta_1) * delta * z_w_b
            v_w_b[:] = beta_2 * v_w_b + (1 - beta_2) * np.power(
                delta * z_w_b, 2
            )
            m_w_b_hat[:] = m_w_b / (1 - beta_1_n)
            v_w_b_hat[:] = v_w_b / (1 - beta_2_n)
            w_b[:] += a_w * m_w_b_hat / (np.sqrt(v_w_b_hat) + eps_adam)

            m_th[:] = beta_1 * m_th + (1 - beta_1) * delta * z_th
            v_th[:] = beta_2 * v_th + (1 - beta_2) * np.power(delta * z_th, 2)
            m_th_hat[:] = m_th / (1 - beta_1_n)
            v_th_hat[:] = v_th / (1 - beta_2_n)
            th[:] += a_th * m_th_hat / (np.sqrt(v_th_hat) + eps_adam)

            m_th_t[:] = beta_1 * m_th_t + (1 - beta_1) * delta * z_th_t
            v_th_t[:] = beta_2 * v_th_t + (1 - beta_2) * np.power(
                delta * z_th_t, 2
            )
            m_th_t_hat[:] = m_th_t / (1 - beta_1_n)
            v_th_t_hat[:] = v_th_t / (1 - beta_2_n)
            th_t[:] += a_th * m_th_t_hat / (np.sqrt(v_th_t_hat) + eps_adam)

            m_th_b = beta_1 * m_th_b + (1 - beta_1) * delta * z_th_b
            v_th_b = beta_2 * v_th_b + (1 - beta_2) * np.power(
                delta * z_th_b, 2
            )
            m_th_b_hat = m_th_b / (1 - beta_1_n)
            v_th_b_hat = v_th_b / (1 - beta_2_n)
            th_b += a_th * m_th_b_hat / (np.sqrt(v_th_b_hat) + eps_adam)

            if a == 0 or a == 1 or t == n_t_max - 1:
                break

            t += 1
            gamma_n *= gamma

            # Adam
            beta_1_n *= beta_1
            beta_2_n *= beta_2

        episodes[iter_no, 0] = mu
        episodes[iter_no, 1] = a
        episodes[iter_no, 2] = r
        episodes[iter_no, 3] = t

        if w_history_episodes is not None and _isin(
            iter_no + 1, w_history_episodes
        ):
            ws[:, :n_taus] = w
            ws[:, n_taus : 2 * n_taus] = w_t
            ws[:, -1] = w_b

            ths[:n_taus] = th
            ths[n_taus : 2 * n_taus] = th_t
            ths[-1] = th_b

            history_episodes.append(iter_no + 1)
            w_history.append(ws.copy())
            th_history.append(ths.copy())

    ws[:, :n_taus] = w
    ws[:, n_taus : 2 * n_taus] = w_t
    ws[:, -1] = w_b

    ths[:n_taus] = th
    ths[n_taus : 2 * n_taus] = th_t
    ths[-1] = th_b

    return ws, ths, episodes, history_episodes, w_history, th_history

# Public interface of _train_with_adam
# For input parameters: see _train_with_adam
#
# Output:
# ws: final signal integrators weights for policy
# ths: final signal integrators weights for value function
# episodes: pandas DataFrame, one row for each training episode;
#   Columns:
#       "mu": mean value of the signal
#       "a": action of the agent at the end of the episode
#       "r": reward delivered
#       "rt": response time / dt (so, an integer)
# w_history: dictionary; key: number of episode during training (in [0, K); they are
#   the values passed in w_history_episodes); value: the 3 x n_taus array containing
#   the weights defining the policy (for the format see ws0 in _train_with_adam)
def train_with_adam(
    ws0,
    ths0,
    taus,
    dt,
    t_max,
    sig,
    sig_i,
    gamma,
    mus,
    rewards=None,
    alpha0=0.001,
    iter_scale=10 ** 4,
    iter_gamma=0.75,
    lambda_min=0.99999,
    lambda_max=1.0 - 1.0 / 100000,
    use_t_integrators=True,
    p3_tot_abs=False,
    beta_1=0.9,
    beta_2=0.999,
    eps_adam=10 ** -8,
    w_history_episodes=None,
    b_exp_i_taus=False,
):

    if rewards is None:
        rewards = np.ones_like(mus)

    try:
        t_max[0]
        t_maxs = t_max
    except TypeError as te:
        t_maxs = t_max * np.ones((min(mus.size, rewards.size),))

    (
        ws,
        ths,
        episodes,
        history_episodes,
        w_history,
        th_history,
    ) = _train_with_adam(
        ws0=ws0,
        ths0=ths0,
        taus=taus,
        dt=dt,
        t_maxs=t_maxs,
        sig=sig,
        sig_i=sig_i,
        gamma=gamma,
        mus=mus,
        rewards=rewards,
        use_t_integrators=use_t_integrators,
        p3_tot_abs=p3_tot_abs,
        alpha0=alpha0,
        iter_scale=iter_scale,
        iter_gamma=iter_gamma,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        beta_1=beta_1,
        beta_2=beta_2,
        eps_adam=eps_adam,
        w_history_episodes=w_history_episodes,
        b_exp_i_taus=b_exp_i_taus,
    )

    w_history = {
        ep: ws for ep, ws, ths in zip(history_episodes, w_history, th_history)
    }

    episodes = pd.DataFrame(data=episodes, columns=["mu", "a", "r", "rt"])
    return ws, ths, episodes, w_history


# Function used to generate episodes (after learning) and compute performance of the agent
# Input parameters: see _train_with_adam and:
# collapse_on_abs_mu: if True, for episodes with mu < 0 the signal becomes -signal and
#   action 0 becomes 1 and 1 becomes 0 (to increase your statistics in case in which
#   you can assume that for an agent episodes with mu and -mu are equivalent - but
#   for the fact that of course "right" -> "left" and "left" -> "right")
# Output: see generate_episodes
@jit(nopython=True)
def _generate_episodes(
    ws,
    taus,
    dt,
    t_maxs,
    sig,
    sig_i,
    gamma,
    mus,
    rewards,
    use_t_integrators=True,
    p3_tot_abs=False,
    collapse_on_abs_mu=False,
    return_signals=True,
    ths=None,
    b_exp_i_taus=False,
):

    n_taus = taus.size
    n_max_t_max = int(np.ceil(t_maxs.max() / dt))
    n_episodes = min(mus.size, rewards.size, t_maxs.size)

    if not b_exp_i_taus:
        i_taus = dt / taus
    else:
        i_taus = 1.0 - np.exp(-dt / taus)

    w = ws[:, :n_taus].copy()
    w_t = ws[:, n_taus : 2 * n_taus].copy()
    w_b = ws[:, -1].copy()

    if ths is not None:
        th = ths[:n_taus]
        th_t = ths[n_taus : 2 * n_taus]
        th_b = ths[-1]
    else:
        th = np.zeros((n_taus,))
        th_t = np.zeros((n_taus,))
        th_b = 0.0

    ints = np.zeros((n_max_t_max, n_taus))
    ints_n = np.zeros((n_max_t_max, n_taus))
    ints_t = np.zeros((n_max_t_max, n_taus))
    ints_t_n = np.zeros((n_max_t_max, n_taus))

    signal_ = np.zeros((n_max_t_max,))
    ps_ = np.zeros((n_max_t_max, 3))
    ps_x = np.zeros_like(ps_)
    vs_ = np.zeros((n_max_t_max,))

    ts = np.arange(0, n_max_t_max)
    for tau_no in range(n_taus):
        ints_t[:, tau_no] = 1.0 - np.exp(-ts * dt / taus[tau_no])

    if ths is not None:
        x = np.zeros((10 ** 6, 4 + 2 * n_taus + 3 + 1))
    else:
        x = np.zeros((10 ** 6, 4 + 2 * n_taus + 3))
    nx = 0
    y = np.zeros((10 ** 5, 6))
    ny = 0

    for episode_no in range(n_episodes):
        mu = mus[episode_no]
        reward = rewards[episode_no]
        t_max = t_maxs[episode_no]
        n_t_max = int(np.ceil(t_max / dt))

        t = 0
        r = 0.0
        v = 0.0
        while t < n_t_max:
            signal = mu + sig * np.random.randn()
            _update_integrators(ints, ints_n, signal, t, i_taus, sig_i)
            if use_t_integrators:
                _update_t_integrators(ints_t, ints_t_n, t, sig_i)

            ps, dummy0 = _compute_ps(
                w, w_t, w_b, ints_n, ints_t_n, t, use_t_integrators, p3_tot_abs
            )
            #             a = np.random.choice(3, size=1, replace=True, p=ps)[0]
            a = np.searchsorted(ps.cumsum(), np.random.rand())

            signal_[t] = signal
            ps_[t, :] = ps

            if ths is not None:
                v, dummy1 = _compute_v(
                    th,
                    th_t,
                    th_b,
                    ints_n,
                    ints_t_n,
                    t,
                    use_t_integrators,
                    p3_tot_abs,
                )

                vs_[t] = v

            if a == 0 and mu > 0.0:
                r = reward
            elif a == 1 and mu < 0.0:
                r = reward

            if a == 0 or a == 1 or t == n_t_max - 1:
                break

            t += 1

        rt = t + 1

        if collapse_on_abs_mu and mu < 0:
            mu = -mu
            signal_ = -signal_
            ints_n[:, :] = -ints_n
            if a == 0:
                a == 1
            elif a == 1:
                a == 0

            ps_x = ps_.copy()
            ps_[:, 0] = ps_x[:, 1]
            ps_[:, 1] = ps_x[:, 0]

        if return_signals:
            while nx + rt > x.shape[0]:
                x0 = x[:nx, :].copy()
                x = np.zeros((int(x.shape[0] * 1.2), x.shape[1]))
                x[:nx, :] = x0
            #             del x0

            x[nx : nx + rt, 0] = episode_no
            x[nx : nx + rt, 1] = np.arange(rt)
            x[nx : nx + rt, 2] = np.arange(rt, 0, -1) - 1
            x[nx : nx + rt, 3] = signal_[:rt]
            x[nx : nx + rt, 4 : 4 + n_taus] = ints_n[:rt, :]
            x[nx : nx + rt, 4 + n_taus : 4 + 2 * n_taus] = ints_t_n[:rt, :]
            x[nx : nx + rt, 4 + 2 * n_taus : 4 + 2 * n_taus + 3] = ps_[:rt, :]

            if ths is not None:
                x[nx : nx + rt, 4 + 2 * n_taus + 3] = vs_[:rt]
            nx += rt

        while ny + 1 > y.shape[0]:
            y0 = y.copy()
            y = np.zeros((int(y.shape[0] * 1.2), y.shape[1]))
            y[:ny, :] = y0
        y[ny, 0] = episode_no
        y[ny, 1] = mu
        y[ny, 2] = n_t_max
        y[ny, 3] = a
        y[ny, 4] = r
        y[ny, 5] = rt
        ny += 1

        episode_no += 1

    x = x[:nx, :]
    y = y[:ny, :]

    return x, y

# Public interface for _generate_episodes
# Input parameters: see _generate_episodes
#
# Output:
# episodes: see train_with_adam
# signals: pandas DataFrame with value of the signal for each time step for each episode;
#   Columns:
#       episode: number of episode
#       t: time step
#       t_back: time step counting from decision time (rt - t)
#       signal: value of the signal at time t
#       int_0, int_1, ...: value of signal integrator 0, 1, ... n_taus-1, corresponding
#           to taus[0], taus[1], etc., at time t
#       int_t0, int_t_1, ...: same as above for time integrators
#       p_0, p_1, p_2: probability of choosing action 0, 1, 2 at time t
def generate_episodes(
    ws,
    taus,
    dt,
    t_max,
    sig,
    sig_i,
    gamma,
    mus,
    rewards=None,
    use_t_integrators=True,
    p3_tot_abs=False,
    collapse_on_abs_mu=False,
    return_signals=True,
    ths=None,
    b_exp_i_taus=False,
):

    if rewards is None:
        rewards = np.ones_like(mus)

    try:
        t_max[0]
        t_maxs = t_max
        b_single_t_max = False
    except (TypeError, IndexError) as e:
        t_maxs = t_max * np.ones((min(mus.size, rewards.size),))
        b_single_t_max = True

    x, y = _generate_episodes(
        ws=ws,
        taus=taus,
        dt=dt,
        t_maxs=t_maxs,
        sig=sig,
        sig_i=sig_i,
        gamma=gamma,
        mus=mus,
        rewards=rewards,
        use_t_integrators=use_t_integrators,
        p3_tot_abs=p3_tot_abs,
        collapse_on_abs_mu=collapse_on_abs_mu,
        return_signals=return_signals,
        ths=ths,
        b_exp_i_taus=b_exp_i_taus,
    )

    n_taus = taus.size
    signal_cols = (
        ["episode", "t", "t_back", "signal"]
        + [f"int_{k}" for k in range(n_taus)]
        + [f"int_t_{k}" for k in range(n_taus)]
        + [f"p_{k}" for k in range(3)]
    )

    if ths is not None:
        signal_cols += ["v"]

    if return_signals:
        signals = pd.DataFrame(data=x, columns=signal_cols)
    else:
        signals = pd.DataFrame(columns=signal_cols)

    signals["episode"] = signals["episode"].astype(int)
    signals["t"] = signals["t"].astype(int)
    signals["t_back"] = signals["t_back"].astype(int)

    episodes = pd.DataFrame(
        data=y, columns=["episode", "mu", "t_max", "a", "r", "rt"]
    )
    episodes["episode"] = episodes["episode"].astype(int)
    episodes["t_max"] = episodes["t_max"].astype(int)
    episodes["a"] = episodes["a"].astype(int)
    episodes["rt"] = episodes["rt"].astype(int)
    episodes["abs_mu"] = np.abs(episodes["mu"])

    if b_single_t_max:
        del episodes["t_max"]

    return episodes, signals


# Function that, given the signals (x, one signal per row), the
# corresponding mus and a threhsold th, computes the single
# integrator performance.
@jit(nopython=True)
def _comp_perf(x, mus, th):
    th = np.abs(th)

    tus = np.zeros((x.shape[0],), dtype=np.int64)
    tds = np.zeros((x.shape[0],), dtype=np.int64)
    for i in range(x.shape[0]):
        tus[i] = np.argmax(x[i, :] >= th)
        tds[i] = np.argmax(x[i, :] <= -th)

    idx = np.where((tus == 0) & (x[:, 0] < th))[0]
    tus[idx] = x.shape[1]

    idx = (tds == 0) & (x[:, 0] > -th)
    tds[idx] = x.shape[1]

    xx = (tus < tds) == (mus.flatten() > 0)

    idx = (tus == x.shape[1]) & (tds == x.shape[1])
    xx[idx] = False

    return xx.sum() / xx.size


# Sampling a Dirichlet distribution
# See: https://stats.stackexchange.com/questions/69210/drawing-from-dirichlet-distribution
@jit(nopython=True)
def _dirichlet(alphas):
    gs = np.zeros_like(alphas)
    for k, alpha in enumerate(alphas):
        gs[k] = np.random.gamma(alpha)
    return gs / gs.sum()


# Function computing (approximate) optimal threshold given a set of
# mus (as in _train_with_adam), a signal sigma and intrinsic noise sigma_i
# for each time-scale in taus (array with n_taus elements).
# Returns:
# perfs: array (n_taus) with performance (estimated with mus_test) for each tau
# ths: array (n_taus) with the estimated thrsholds
@jit(nopython=True)
def _comp_th(
    mus,
    mus_test,
    sig,
    sig_i_eff,
    taus,
    dt,
    t_max,
    nz=10 ** 3,
    b_exp_i_taus=False,
):
    n_taus = taus.size
    n_t_max = int(np.ceil(t_max / dt))

    if not b_exp_i_taus:
        i_taus = dt / taus
    else:
        i_taus = 1.0 - np.exp(-dt / taus)

    n_episodes = mus.size
    n_test_episodes = mus_test.size

    signals = mus.reshape(n_episodes, 1) + sig * np.random.randn(
        mus.size, n_t_max
    )

    signals_test = mus_test.reshape(
        n_test_episodes, 1
    ) + sig * np.random.randn(mus_test.size, n_t_max)

    x = np.zeros((n_episodes, n_t_max + 1))
    x_up = np.zeros_like(x)
    x_dw = np.zeros_like(x)
    x_test = np.zeros((n_test_episodes, n_t_max + 1))

    perfs = np.zeros((n_taus,))
    ths = np.zeros((n_taus,))
    for tau_no, i_tau in enumerate(i_taus):
        x[:, 0] = 0.0
        x_test[:, 0] = 0.0
        for t in range(1, n_t_max + 1):
            x[:, t] = x[:, t - 1] - (x[:, t - 1] - signals[:, t - 1]) * i_tau
            x_test[:, t] = (
                x_test[:, t - 1]
                - (x_test[:, t - 1] - signals_test[:, t - 1]) * i_tau
            )

        x[:, 1:] += sig_i_eff * np.random.randn(x.shape[0], x.shape[1] - 1)
        x_test[:, 1:] += sig_i_eff * np.random.randn(
            x_test.shape[0], x_test.shape[1] - 1
        )

        for i in range(n_episodes):
            if mus[i] < 0.0:
                x[i, :] *= -1

        for t in range(1, n_t_max + 1):
            x_up[:, t] = np.maximum(x_up[:, t - 1], x[:, t])
            x_dw[:, t] = np.minimum(x_dw[:, t - 1], x[:, t])
        x_dw *= -1

        dz = np.abs(x).max() / (nz - 10 ** -4)
        z = np.arange(nz + 1) * dz
        cz = np.zeros((nz + 1,), dtype=np.int64)
        pz = np.zeros((nz + 1,))

        for i in range(n_episodes):
            is_up = x_up[i, :] > x_dw[i, :]
            is_up[-1] = False
            b_starts = np.where((is_up[1:]) & (~is_up[:-1]))[0] + 1
            b_ends = np.where((~is_up[1:]) & (is_up[:-1]))[0] + 1

            for s, e in zip(b_starts, b_ends):
                k_dw = np.int64(x_dw[i, s] / dz + 1)
                k_up = np.int64(x_up[i, e] / dz - 0)
                cz[k_dw:k_up] += 1

        ths_dirichl = np.zeros((100,))
        for k in range(ths_dirichl.size):
            pz[:] = _dirichlet(cz + 0.5)
            n = pz.argmax()
            ths_dirichl[k] = 0.5 * (z[n] + z[n + 1])

        th = ths_dirichl.mean()
        n = (z >= th).argmax()

        perf = _comp_perf(x_test[:, 1:], mus_test, th)
        perf_on_x = cz[n] / n_episodes

        perfs[tau_no] = perf
        ths[tau_no] = th

    return perfs, ths

# Utility function: takes a list p_mu_params describing
# the distribution of mu (for example, Gaussian) and returns
# a function that samples from the distribution
def get_p_mu(p_mu_params):
    if p_mu_params[0].lower() == "uniform":
        p_mu = lambda size=None: np.random.uniform(
            p_mu_params[1], p_mu_params[2], size=size
        )
    elif p_mu_params[0].lower() == "gaussian":
        p_mu = (
            lambda size=None: p_mu_params[1]
            + p_mu_params[2] * np.random.randn(size)
            if size is not None and size != 1
            else p_mu_params[1] + p_mu_params[2] * np.random.randn()
        )

    elif p_mu_params[0].lower() == "delta":
        if len(p_mu_params) > 2:
            p = p_mu_params[2]
        else:
            p = 0.5

        p_mu = (
            lambda size=None: p_mu_params[1]
            * (2 * (np.random.rand(size) > p) - 1)
            if size is not None and size != 1
            else p_mu_params[1] * (2 * (np.random.rand() > p) - 1)
        )
    elif p_mu_params[0].lower() == "choice":
        p_mu = lambda size=None: np.random.choice(
            p_mu_params[1], p=p_mu_params[2], size=size
        )
    elif p_mu_params[0].lower() == "simple_gauss_pretrain":

        def _p_mu(n, m_mu, sig_mu, n_pre_train):
            if n is None:
                return m_mu + sig_mu * np.random.randn()
            else:
                mus = np.zeros(n)
                for k in range(n):
                    if k < n_pre_train:
                        mus[k] = (
                            m_mu
                            + (2.0 * (np.random.rand() > 0.5) - 1.0) * sig_mu
                        )
                    else:
                        mus[k] = m_mu + sig_mu * np.random.randn()

                return mus

        p_mu = lambda size=None: _p_mu(
            size, p_mu_params[1], p_mu_params[2], p_mu_params[3]
        )

    return p_mu

# Utility function: takes a list p_t_max_params describing
# the distribution of t_max (maximum duration of an episode; it could be
# deterministic, for example - as in the paper) and returns
# a function that samples from the distribution
def get_p_t_max(p_t_max_params):
    if p_t_max_params[0].lower() == "uniform":
        p_t_max = lambda size=None: np.random.uniform(
            p_t_max_params[1], p_t_max_params[2], size=size
        )
    elif p_t_max_params[0].lower() == "deterministic":
        p_t_max = (
            lambda size=None: p_t_max_params[1] * np.ones((size,))
            if size is not None and size != 1
            else p_t_max_params[1]
        )
    elif p_t_max_params[0].lower() == "choice":
        p_t_max = lambda size=None: np.random.choice(
            p_t_max_params[1], p=p_t_max_params[2], size=size
        )

    return p_t_max

# Utility function: takes a list p_r_params describing
# the distribution of reward for a correct response on an
# episode with signal mean mu, and returns a function p_r(mu)
# sampling from the distribution (throughout the paper, p_r(mu) = 1)
def get_p_r(p_r_params):
    def is_float(potential_float):
        try:
            float(potential_float)
            return True
        except Exception:
            return False

    if p_r_params[0].lower() == "one":
        p_r = lambda mus: 1.0 if is_float(mus) else np.ones_like(mus)
    elif p_r_params[0].lower() == "exponential":
        p_r = lambda mus: np.exp(
            -np.abs(mus) * p_r_params[2] / p_r_params[1]
        ) / (
            np.exp(0.5 * p_r_params[2] ** 2)
            * special.erfc(p_r_params[2] / np.sqrt(2))
        )

    return p_r

# Main training function. In train_with_adam you have to specify, among others,
# the value of mu for each episode, and the values of the timescale taus.
# Here you pass the distribution of mu (and of t_max and reward; see utilities above),
# the number of taus n_taus you want and the number of training episodes, and that's it:
# the function performs the training for that specific case.
#
# Input parameters: see _train_with_adam, and...
# p_mu_params, p_t_max_params, p_r_params: see above get_p_mu, get_p_t_max, and get_p_r
# n_training_steps: number of training steps
# n_test_steps: number of episodes to generate at the end of training
# n_taus, tau_min, tau_max, tau_distr: you want n_taus time scales taus,
#   from tau_min to tau_max (included) in logarithmic (if tau_distr="exp", as in the paper)
#   or linear (if tau_distr="lin") scale.
# p0_choice: at the beginning of training you want the agent to make a
#   (random) "right" or "left" decision with this probability
#   (in order not to make the agent wait forever)
# n_history, n_history_per_decade: they deterimine w_history_episodes (as in _train_with_adam);
#   if n_history is not None, w_history_episodes has n_history elements that cover
#   the interval [0, n_training_steps) in logarithmic scale;
#   if n_history_per_decade is not None, w_history_episodes covers the [0, n_training_steps) interval
#   in logarithmic scale with (approximately) n_history_per_decade points for each decade.
#
# Output:
# ws, ths, training_outputs, w_history: see train_with_adam (training_outputs=episodes, weights_history=w_history)
# perform: (average reward, average rt)
#  perform_single_tau = pandas DataFrame with performance for single integrators;
#   Columns:
#       tau: the time scale of the integrator (tau in taus)
#       rep: number of repetition of the optimization (see _comp_th)
#       correct: performance as fraction of correct episodes
#       threshold: computed threshold (see _comp_th)
# sig_i_eff: = sig_i * alpha_i as defined in Eq. 18
# train_agent_data: a dictionary with the parameters used for the training  
def train_agent(
    sig,
    p_mu_params,
    p_t_max_params,
    sig_i,
    n_training_steps,
    n_test_steps,
    use_t_integrators=True,
    p3_tot_abs=False,
    p_r_params=["one"],
    n_taus=10,
    tau_min=10 ** -1,
    tau_max=10 ** 1,
    dt=10 * 10 ** -3,
    gamma=np.exp(np.log(0.5) / 10 ** 5),
    p0_choice=1.0 / 50.0,
    alpha0=0.001,
    iter_scale=10 ** 4,
    iter_gamma=0.75,
    n_history=None,
    n_history_per_decade=20,
    verbose=True,
    b_exp_i_taus=False,
    tau_distr="exp",
    ws0=None,
    ths0=None,
):
    tic_toc = TicToc()

    if tau_distr.lower() == "exp":
        taus = np.exp(np.linspace(np.log(tau_min), np.log(tau_max), n_taus))
    elif tau_distr.lower() == "lin":
        taus = np.linspace(tau_min, tau_max, n_taus)

    if callable(p_mu_params):
        p_mu = p_mu_params
    elif type(p_mu_params) != list:
        p_mu = get_p_mu(["Gaussian", 0.0, p_mu_params])
    elif p_mu_params[0].lower() == "gaussian" and p_mu_params[1] == 0.0:
        p_mu = get_p_mu(p_mu_params)
    else:
        p_mu = get_p_mu(p_mu_params)

    p_t_max = get_p_t_max(p_t_max_params)

    p_r = get_p_r(p_r_params)

    train_agent_data = {
        "sig": sig,
        "p_mu_params": p_mu_params,
        "p_t_max_params": p_t_max_params,
        "p_r_params": p_r_params,
        "sig_i": sig_i,
        "n_training_steps": n_training_steps,
        "n_test_steps": n_test_steps,
        "use_t_integrators": use_t_integrators,
        "p3_tot_abs": p3_tot_abs,
        "dt": dt,
        "taus": taus,
        "gamma": gamma,
        "p0_choice": p0_choice,
        "alpha0": alpha0,
        "iter_scale": iter_scale,
        "iter_gamma": iter_gamma,
        "b_exp_i_taus": b_exp_i_taus,
    }

    if ws0 is None:
        ws = 10 ** -3 * np.random.randn(3, 2 * n_taus + 1)
        ws[0, -1] = np.log(p0_choice / (1 - p0_choice) / 2)
        ws[1, -1] = ws[0, -1]
        ws[2, -1] = 0.0
    else:
        ws = ws0.copy()

    if not use_t_integrators:
        ws[:, n_taus : (2 * n_taus)] = 0.0

    if ths0 is None:
        ths = 10 ** -3 * np.random.randn(2 * n_taus + 1)
        ths[-1] = 0.5
    else:
        ths = ths0.copy()

    mus = p_mu(n_training_steps)
    rewards = p_r(mus)
    t_maxs = np.array([p_t_max() for k in range(n_training_steps)])

    if n_history is not None:
        n_history_ = n_history
    elif n_history_per_decade is None:
        n_history_ = 100
    else:
        n_history_ = (
            int(np.floor(np.log10(n_training_steps))) * n_history_per_decade
        )
        n_history_ = max(n_history_, min(10, n_training_steps))

    w_history_episodes = (
        np.unique(
            np.round(
                10
                ** np.linspace(
                    np.log10(1.0), np.log10(n_training_steps + 1), n_history_
                )
            ).astype(int)
        )
        - 1
    )

    ws, ths, training_outputs, weights_history = train_with_adam(
        ws,
        ths,
        taus,
        dt,
        t_maxs,
        sig,
        sig_i,
        gamma,
        mus,
        rewards,
        alpha0=alpha0,
        iter_scale=iter_scale,
        iter_gamma=iter_gamma,
        lambda_min=1.0 - 1.0 / 10 ** 5,
        lambda_max=1.0 - 1.0 / 10 ** 5,
        use_t_integrators=use_t_integrators,
        p3_tot_abs=p3_tot_abs,
        beta_1=0.9,
        beta_2=0.999,
        w_history_episodes=w_history_episodes,
    )

    mus = p_mu(n_test_steps)
    rewards = p_r(mus)
    t_maxs = np.array([p_t_max() for k in range(n_test_steps)])
    eps, _ = generate_episodes(
        ws,
        taus,
        dt,
        t_maxs,
        sig,
        sig_i,
        gamma,
        mus,
        rewards,
        use_t_integrators=use_t_integrators,
        p3_tot_abs=p3_tot_abs,
        return_signals=False,
    )

    perform = (eps.r.mean(), eps.rt.mean())

    elapsed = tic_toc.toc(False)

    if verbose:
        print(
            f"sig_i={sig_i:1.2f}: training_steps={n_training_steps}, correct={perform[0]*100.:1.1f}%, <rt>={perform[1] * dt * 1000:1.1f} ms, elapsed {elapsed:.1f} s"
        )

    wx = ws[0:1, :n_taus].mean(axis=0)
    wx = wx / wx.max()
    sig_i_eff = sig_i / np.sqrt((wx ** 2).sum())
    del wx

    # Single-tau performance
    if True:
        n_reps = 10
        t_max = np.array([p_t_max() for k in range(10 ** 6)]).mean()
        perf_data = np.zeros((n_taus * n_reps, 4))
        perf_data[:, 0] = np.tile(taus, n_reps)
        perf_data[:, 1] = np.repeat(np.arange(n_taus), n_reps)
        for j in range(n_reps):
            mus = p_mu(10 ** 4)
            mus_test = p_mu(10 ** 4)
            perfs, thresholds = _comp_th(
                mus=mus,
                mus_test=mus_test,
                sig=sig,
                sig_i_eff=sig_i_eff,
                taus=taus,
                dt=dt,
                t_max=t_max,
                nz=10 ** 3,
                b_exp_i_taus=b_exp_i_taus,
            )
            perf_data[j * n_taus : (j + 1) * n_taus, 2] = perfs
            perf_data[j * n_taus : (j + 1) * n_taus, 3] = thresholds
        del perfs, thresholds, j

        perform_single_tau = pd.DataFrame(
            data=perf_data, columns=["tau", "rep", "correct", "threshold"]
        )
        del perf_data
        perform_single_tau["rep"] = perform_single_tau["rep"].astype(int)
    else:
        perform_single_tau = None

    return (
        ws,
        ths,
        weights_history,
        training_outputs,
        perform,
        perform_single_tau,
        sig_i_eff,
        train_agent_data,
    )

# In Roitman JD, Shadlen MN "Response of neurons in the lateral intraparietal
# area during a combined visual discrimination reaction time task", they have
# a screen refresh rate of 75 Hz and the dots are moved on the screen every
# 3 frames - just a handy conversion factor used to link our si
shadlen_dt = (75 / 3) ** -1

# It takes mu_norm (that is: mu / sigma_mu) and coherence_to_mu_factor
# (the only fitted value in converting signal mean mu to coherence as
# defined in Roitman JD, Shadlen MN)
def coherence_from_mu(mu_norm, coherence_to_mu_factor=0.216):
    if type(mu_norm) == list:
        mu_norm = np.array(mu_norm)

    x0 = coherence_to_mu_factor
    return (
        -(mu_norm ** 2) + np.sqrt(mu_norm ** 4 + 400 * mu_norm ** 2 * x0 ** 2)
    ) / (2 * x0 ** 2)


# Inverse function of coherence_from_mu
def mu_from_coherence(c, coherence_to_mu_factor=0.216):
    if type(c) == list:
        c = np.array(c)
    return coherence_to_mu_factor * c / np.sqrt(100.0 - c)

# Utility function (to move one column before another one in a pandas DataFrame)
def move_col_before(df_in, col, before):
    df = df_in.copy()
    column = df.pop(col)
    df.insert(
        df.columns.get_loc(before), column.name, column, allow_duplicates=False
    )
    return df


# Computes the "Shadlen-like" plots (Fig. 4 in the paper)
# Input:
# episodes, signals, ws: see generate_episodes
# dt, p3_tot_abs: see _train_with_adam
# n_t_max: maximum duration of episodes (in steps of duration dt)
# start_to_percentile: when aligning to decision time, the curves
#   will go back for a lapse of time equal to the percentile
#   start_to_percentile of the response times
# end_to_percentile: when aligning to start of episode, we'll look
#   forward until end_to_percentile percentile of the response times
# start_to_n_samples: when aligning to decision time, the curves
#   will average at least a start_to_n_samples episodes
#   (the more you go back in time, the fewer episodes you are left with,
#   since you are requiring longer and longer response times)
# end_to_n_samples: when aligning to start of episode, we go forward in time
#   in the episode until fewer than end_to_n_samples episodes remains;
#   then we stop
# Returns:
# align2start, align2decision: two pandas DataFrames (one for statistics
#   when aligning to begin of episode; the other when aligning to decision
#   time) with columns that store mean and standard deviation for the
#   signal, the integrators (see generate_episodes for the naming convention),
#   and DeltaSigmaS and DeltaSigmaT (Eq. 23 and 24 in the paper; note that
#   in the final version of the text, DeltaSigmaT becomes DeltaSigmaC)
def compute_shadlen(
    episodes,
    signals,
    ws,
    dt=0.01,
    n_t_max=None,
    p3_tot_abs=False,
    start_to_percentile=50.0,
    end_to_percentile=None,
    start_to_n_samples=None,
    end_to_n_samples=None,
):
    def percentile(n):
        def percentile_(x):
            return np.percentile(x, n)

        percentile_.__name__ = "percentile_%s" % n
        return percentile_

    if "t_max" in episodes:
        y = signals.merge(
            episodes[
                (episodes["rt"] < episodes["t_max"]) | (episodes["a"] != 2)
            ],
            on="episode",
            how="right",
        )
    else:
        if n_t_max is None:
            n_t_max = signals["t"].max()
        y = signals.merge(
            episodes[(episodes["rt"] < n_t_max) | (episodes["a"] != 2)],
            on="episode",
            how="right",
        )
    y["success"] = (y["r"] > 0.0).astype(int)
    episodes["success"] = (episodes["r"] > 0.0).astype(int)

    n_taus = (ws.shape[1] - 1) // 2

    # Compute DeltaSigmas
    ints = y[["int_" + str(k) for k in range(n_taus)]].values
    y["DeltaSigmaS"] = ws[0, :n_taus] @ ints.T
    if p3_tot_abs:
        y["DeltaSigmaS"] -= np.abs(ws[2, :n_taus] @ ints.T)
    else:
        y["DeltaSigmaS"] -= ws[2, :n_taus] @ np.abs(ints).T

    ints = y[["int_t_" + str(k) for k in range(n_taus)]].values
    y["DeltaSigmaT"] = (
        ws[0, n_taus : 2 * n_taus] - ws[2, n_taus : 2 * n_taus]
    ) @ ints.T

    y["DeltaSigma"] = (
        y["DeltaSigmaS"] + y["DeltaSigmaT"] + ws[0, -1] - ws[2, -1]
    )
    del ints

    for k in range(n_taus):
        y["abs_int_" + str(k)] = y["int_" + str(k)].abs()

    y = move_col_before(y, "DeltaSigma", "int_0")
    y = move_col_before(y, "DeltaSigmaS", "int_0")
    y = move_col_before(y, "DeltaSigmaT", "int_0")

    def f(c):
        if c == "int_0":
            return ("mean", "count")
        if c.startswith("DeltaSigma"):
            return ("mean", "std")
        else:
            return "mean"

    if end_to_percentile is not None and end_to_percentile < 100.0:
        align2decision = y.merge(
            episodes.groupby(["mu", "a", "success"])
            .agg({"rt": percentile(end_to_percentile)})
            .reset_index()
            .rename(columns={"rt": "max_rt"}),
            on=["mu", "a", "success"],
        )
        align2decision = align2decision[
            (align2decision["t_back"] < align2decision["max_rt"])
            & (align2decision["t"] * dt * 1000.0 >= 0)
        ]
        del align2decision["max_rt"]
    else:
        align2decision = y.copy()

    align2decision = (
        align2decision.groupby(["t_back", "mu", "success", "a"])
        .agg(
            {
                c: f(c)
                for c in align2decision.columns
                if c[:3] == "int"
                or c == "r"
                or c[: len("DeltaSigma")] == "DeltaSigma"
                or c[: len("abs_int_")] == "abs_int_"
            }
        )
        .reset_index()
        .rename(columns={"t_back": "t"})
        .sort_values(["mu", "success", "t"], ascending=[False, False, True])
        .reset_index(drop=True)
    )

    # Roitman&Shadlen 2002
    # https://www.jneurosci.org/content/jneuro/22/21/9475.full.pdf
    # "The response averages shown in the left half of Figure 7A are drawn to
    # the median reaction time and do not include any activity in the 100 msec preceding saccade initiation"
    if start_to_percentile is not None and start_to_percentile < 100.0:
        align2start = y.merge(
            episodes.groupby(["mu", "a", "success"])
            .agg({"rt": percentile(start_to_percentile)})
            .reset_index()
            .rename(columns={"rt": "max_rt"}),
            on=["mu", "a", "success"],
        )
        align2start = align2start[
            (align2start["t"] < align2start["max_rt"])
            & (align2start["t_back"] * dt * 1000.0 >= 100.0)
        ]
        del align2start["max_rt"]
    else:
        align2start = y.copy()
        align2start = align2start[align2start["t_back"] * dt * 1000.0 >= 100.0]

    align2start = (
        align2start.groupby(["t", "mu", "success", "a"])
        .agg(
            {
                c: f(c)
                for c in align2start.columns
                if c[:3] == "int"
                or c == "r"
                or c[: len("DeltaSigma")] == "DeltaSigma"
                or c[: len("abs_int_")] == "abs_int_"
            }
        )
        .reset_index()
        .sort_values(["mu", "success", "t"], ascending=[False, False, True])
        .reset_index(drop=True)
    )

    del episodes["success"]

    def f(c):
        if c[1] == "mean" or len(c[1]) == 0:
            s = c[0]
        elif c[1] == "std":
            s = c[0] + "_std"
        else:
            s = c[1]
        return s

    align2decision.columns = [
        f(c)
        for c in zip(
            align2decision.columns.get_level_values(0),
            align2decision.columns.get_level_values(1),
        )
    ]
    align2decision = align2decision[
        [c for c in align2decision.columns if c[:3] != "int"]
        + [c for c in align2decision.columns if c[:3] == "int"]
    ]
    if end_to_n_samples is not None and end_to_n_samples > 0:
        align2decision = align2decision[
            align2decision["count"] >= end_to_n_samples
        ]

    align2start.columns = [
        f(c)
        for c in zip(
            align2start.columns.get_level_values(0),
            align2start.columns.get_level_values(1),
        )
    ]
    align2start = align2start[
        [c for c in align2start.columns if c[:3] != "int"]
        + [c for c in align2start.columns if c[:3] == "int"]
    ]
    if start_to_n_samples is not None:
        align2start = align2start[align2start["count"] >= start_to_n_samples]
    del f

    return align2start, align2decision

# Computes the metrics introduced in the paper to measure signal neutrality
# (see subsection "Signal neutrality and scalar property measures" in the paper)
# episodes, signals, ws: see generate_episodes
# train_agent_data: see train_agent
# max_delta_t: compute the max of the distance between the curves on a period
#   that goes from decision time back to max_delta_t steps
# end_to_n_samples: see compute_shadlen
def compute_shadlen_dist(
    episodes,
    signals,
    ws,
    train_agent_data,
    max_delta_t=60,
    end_to_n_samples=100,
):
    def percentile(n):
        def percentile_(x):
            return np.percentile(x, n)

        percentile_.__name__ = "percentile_%s" % n
        return percentile_

    dt = train_agent_data["dt"]
    taus = train_agent_data["taus"]
    t_max = train_agent_data["p_t_max_params"][1]
    sig = train_agent_data["sig"]
    sig_i = train_agent_data["sig_i"]
    gamma = train_agent_data["gamma"]
    use_t_integrators = train_agent_data["use_t_integrators"]
    p3_tot_abs = train_agent_data["p3_tot_abs"]
    b_exp_i_taus = train_agent_data["b_exp_i_taus"]

    _, align2decision = compute_shadlen(
        episodes,
        signals,
        ws,
        dt=dt,
        n_t_max=None,
        p3_tot_abs=p3_tot_abs,
        start_to_percentile=50.0,
        end_to_n_samples=end_to_n_samples,
    )

    n_episodes_per_mu = int(episodes.shape[0] / episodes["mu"].unique().size)

    x = align2decision.copy()

    x = x[(x["success"] == 1) & (x["a"] == 0)]
    x = x[[c for c in x.columns if not "int_" in c]]
    del x["success"], x["a"], x["r"], x["DeltaSigmaS"]
    del x["DeltaSigmaS_std"], x["DeltaSigmaT"], x["DeltaSigmaT_std"]

    del x["count"]

    if x.shape[0] > 0:
        delta_t = x["t"].max() + 1
        if delta_t > max_delta_t:
            x = x[x["t"] < max_delta_t]
            delta_t = max_delta_t

    if x.shape[0] > 0:
        x["DeltaSigma_norm"] = (x["DeltaSigma"] - x["DeltaSigma"].min()) / (
            x["DeltaSigma"].max() - x["DeltaSigma"].min()
        )

        def f(g):
            d = {}
            d["DeltaSigma_diff"] = (
                g["DeltaSigma_norm"].max() - g["DeltaSigma_norm"].min()
            )
            d["count"] = g.shape[0]
            return pd.Series(d, dtype=np.object)

        x = x.groupby("t").apply(f).reset_index()
        del f
        x = x[x["count"] > 1]

    if x.shape[0] > 0:
        delta_t = x["t"].max() + 1
        delta_sigma_diff_max = x["DeltaSigma_diff"].max()
        delta_sigma_diff_mean = x["DeltaSigma_diff"].mean()
    else:
        delta_t = 0
        delta_sigma_diff_max = np.inf
        delta_sigma_diff_mean = np.inf

    return delta_sigma_diff_mean, delta_sigma_diff_max, delta_t

# Given the weights ws (see generate_episodes and train_agent), the function
# generate the epidosed, compute the "Shadlen" curves, and compute the
# signal neutrality measure.
# Input:
# ws, train_agent_data: see train_agent
# n_episodes_performance: performance of the agent computed on this number of episodes
# n_episodes_per_mu: for each mean value of the signal/for each value of the coherence
#   (see also next coherences), the function generates n_episodes_per_mu episodes (to
#   use to compute the signal neutrality distance)
# coherences: the set of coherence to use to evaluate signal neutrality
# coherence_to_mu_factor: see mu_from_coherence
# 
# Output:
# res: pandas DataFrame, with columns:
#   delta_sigma_diff_mean: for each time step we take the maximum of the
#       distance between the DeltaSigmas for each pair of coherence;
#       this is the average of this distance over all the time steps. This
#       is what is used in the paper
#   delta_sigma_diff_max: as delta_sigma_diff_mean, but in the end we take the maximum
#   delta_t: the actual time period (in time steps) over which the delta_sigma_diff are computed
#   p: fraction of correct decision ([0, 1])
#   cv_mean: for each mu/coherence we compute the coefficient of variation (CV)
#       of the response times; cv_mean is the mean over the mus/coherences
#   cv_min: as cv_mean, but the minimum value is taken (used to compute the scalar property measure)
#   cv_max: as cv_mean, but the maximum value is taken (used to compute the scalar property measure)
def compute_shadlen_history_1step(
    ws,
    train_agent_data,
    n_episodes_performance=10 ** 4,
    n_episodes_per_mu=10 ** 3,
    coherences=[10 ** -3, 3.2, 6.4, 12.8, 25.6, 51.2],
    coherence_to_mu_factor=0.216,
):

    dt = train_agent_data["dt"]
    taus = train_agent_data["taus"]
    t_max = train_agent_data["p_t_max_params"][1]
    sig = train_agent_data["sig"]
    sig_i = train_agent_data["sig_i"]
    gamma = train_agent_data["gamma"]
    use_t_integrators = train_agent_data["use_t_integrators"]
    p3_tot_abs = train_agent_data["p3_tot_abs"]
    b_exp_i_taus = train_agent_data["b_exp_i_taus"]

    if "p_r_params" in train_agent_data:
        reward_f = lambda mus: td_jit.get_p_r(train_agent_data["p_r_params"])(
            mus
        )
    else:
        reward_f = lambda mus: td_jit.get_p_r(["one"])(mus)
    sig_mu = train_agent_data["p_mu_params"][2]
    # print(sig_mu)

    mus = sig_mu * np.random.randn(n_episodes_performance)
    rewards = reward_f(mus)

    episodes, _ = td_jit.generate_episodes(
        ws,
        taus=taus,
        dt=dt,
        t_max=t_max,
        sig=sig,
        sig_i=sig_i,
        gamma=gamma,
        mus=mus,
        rewards=rewards,
        use_t_integrators=use_t_integrators,
        p3_tot_abs=p3_tot_abs,
        collapse_on_abs_mu=False,
        return_signals=False,
        b_exp_i_taus=b_exp_i_taus,
    )
    del mus, rewards

    p = episodes[episodes["r"] > 0].shape[0] / episodes.shape[0]
    p_sem = np.sqrt(p * (1.0 - p) / episodes.shape[0])

    mus = (
        sig
        * np.sqrt(dt / td_jit.shadlen_dt)
        * np.random.choice(
            mu_from_coherence(
                coherences, coherence_to_mu_factor=coherence_to_mu_factor
            ),
            len(coherences) * n_episodes_per_mu,
        )
    )
    rewards = reward_f(mus)
    episodes, signals = td_jit.generate_episodes(
        ws,
        taus=taus,
        dt=dt,
        t_max=t_max,
        sig=sig,
        sig_i=sig_i,
        gamma=gamma,
        mus=mus,
        rewards=rewards,
        use_t_integrators=use_t_integrators,
        p3_tot_abs=p3_tot_abs,
        collapse_on_abs_mu=False,
        return_signals=True,
        b_exp_i_taus=b_exp_i_taus,
    )
    del mus, rewards

    (
        delta_sigma_diff_mean,
        delta_sigma_diff_max,
        delta_t,
    ) = compute_shadlen_dist(
        episodes,
        signals,
        ws,
        train_agent_data,
        max_delta_t=60,
        end_to_n_samples=100,
    )

    g = episodes.copy()
    g = g[g["r"] > 0.0]
    g = g.groupby(["mu", "a"]).agg({"rt": ("mean", "std")}).reset_index()
    g[("rt", "cv")] = g[("rt", "std")] / g[("rt", "mean")]
    g = g.sort_values("mu")
    cv_mean = g[("rt", "cv")].mean()
    cv_max = g[("rt", "cv")].max()
    cv_min = g[("rt", "cv")].min()
    cv_mag_min_ratio = cv_max / cv_min
    del g

    res = {
        "delta_sigma_diff_mean": delta_sigma_diff_mean,
        "delta_sigma_diff_max": delta_sigma_diff_max,
        "delta_t": delta_t,
        "p": p,
        "cv_mean": cv_mean,
        "cv_min": cv_min,
        "cv_max": cv_max,
    }

    return res

# Main function introduces the default parameter values used in the paper;
# it performs training (calling train_agent); generates some episodes (calling
# generate_episodes) with the same distribution of mu/coherence as in training
# phase (Gaussian) and shows the returned episodes DataFrame (presenting a bunch
# of statistics grouping by action taken by the agent). Then it generates episodes
# with mu/coherence extracted by a finite set and calls compute_shadlen_dist; finally
# it calls compute_shadlen to produce the curves (DataFrames) for the plot à la Shadlen.
# See also the companion Jupyter Notebook.
def main():
    # ~2 minutes on am i9 (with Numba)
    # Main parameters used in the paper
    dt = 10 * 10 ** -3
    sig = 1.8  # note that the signal standard deviation reported in the paper is 0.18, that is sig * sqrt(dt)
    sig_mu = 0.25
    sig_i = 0.02
    gamma_ = np.exp(np.log(0.5) / 10 ** 6)  # discount factor, ~1
    use_t_integrators = (
        True  # False only if you don't want to use the time-integrators
    )
    p3_tot_abs = (
        False  # This must be False (no other use with reference to the paper)
    )

    alpha0 = 10 ** -3  # learning rate
    iter_gamma = 0.75
    iter_scale = 10 ** 4

    n_taus = 10
    tau_min = 0.1
    tau_max = 10.0
    b_exp_i_taus = False
    tau_distr = "exp"

    t_max = 2.0

    p_mu_params = ["Gaussian", 0.0, sig_mu]

    n_training_steps = 100 * 10 ** 3
    n_test_steps = 10 ** 5

    (
        ws,
        ths,
        weights_history,
        training_outputs,
        perform,
        perform_single_tau,
        sig_i_eff,
        train_agent_data,
    ) = train_agent(
        sig=sig,
        p_mu_params=p_mu_params,
        p_t_max_params=["deterministic", t_max],
        sig_i=sig_i,
        n_training_steps=n_training_steps,
        n_test_steps=n_test_steps,
        use_t_integrators=use_t_integrators,
        p3_tot_abs=p3_tot_abs,
        dt=dt,
        gamma=gamma_,
        n_taus=n_taus,
        tau_min=tau_min,
        tau_max=tau_max,
        alpha0=alpha0,
        iter_scale=iter_scale,
        iter_gamma=iter_gamma,
        n_history_per_decade=40,
        b_exp_i_taus=b_exp_i_taus,
        tau_distr=tau_distr,
    )

    print(
        f"There are {n_taus} time scales, between {tau_min:1.1f} s and {tau_max:1.0f} s."
    )
    print(
        f"ws (ws.shape = {ws.shape}) is the set of weights at the end of training."
    )
    print(
        f"\tColumns = {{right, left, wait}}; rows: integrators ({n_taus}), time integrators ({n_taus}), biases (1)"
    )
    print(
        f"ths are the weights for the value function estimation ('critic' in actor-critic learning)."
    )
    print(
        f"weights_history is a dictionary logging the weights during a sample of training steps."
    )
    print(
        f"training_outputs is a Pandas DataFrame recording all the mu values,"
    )
    print(
        f"\tthe final action (0=right, 1=left, 2=wait), the reward delivered (r),"
    )
    print(
        f"\tand the reaction time (rt, number of time steps) for each training trial."
    )
    print(
        f"perform is a 2-ple: average reward obtained and average response time (number of time steps) at the end of training."
    )

    print(
        f"perform_single_tau is the result of the single-time scale integrator threshold optimization"
    )
    print(
        f"\t(10 repetitions of the 'stochastic' optimization for each of the taus)"
    )
    print(
        f"sig_i_eff is the intrinsic noise level used for the single-time scale integrator (see Methods - Rescaling sig_i)."
    )
    print(
        f"train_agent_data is a dictionary collecting all the parameters used for training."
    )

    # Generating a number of episodes with the same statistics for mu used during training
    # to collect a few stats on the performance and the response times for different actions
    sig_mu = train_agent_data["p_mu_params"][2]

    mus = sig_mu * np.random.randn(10 ** 5)

    if "p_r_params" in train_agent_data:
        rewards = get_p_r(train_agent_data["p_r_params"])(mus)
    else:
        rewards = get_p_r(["one"])(mus)

    episodes, _ = generate_episodes(
        ws,
        taus=train_agent_data["taus"],
        dt=train_agent_data["dt"],
        t_max=train_agent_data["p_t_max_params"][1],
        sig=train_agent_data["sig"],
        sig_i=train_agent_data["sig_i"],
        gamma=train_agent_data["gamma"],
        mus=mus,
        rewards=rewards,
        use_t_integrators=train_agent_data["use_t_integrators"],
        p3_tot_abs=train_agent_data["p3_tot_abs"],
        collapse_on_abs_mu=False,
        return_signals=False,
        b_exp_i_taus=train_agent_data["b_exp_i_taus"],
    )

    print("This is a sample of episodes...")
    print(episodes.head())

    def f(g):
        d = {}

        d["correct %"] = g["r"].mean()
        idxs = g["r"] > 0.0
        rts = g["rt"].loc[idxs].values
        if rts.size > 0:
            d["RT (correct)"] = rts.mean()
            d["CV_RT (correct)"] = rts.std() / rts.mean()
            d["RT 5%"] = np.percentile(rts, 5)
            d["RT 95%"] = np.percentile(rts, 95)
            d["N"] = rts.size
        else:
            d["RT (correct)"] = None
            d["CV_RT (correct)"] = None
            d["RT 5%"] = None
            d["RT 95%"] = None
            d["N"] = 0

        return pd.Series(d, dtype=np.object)

    print(f"These are stats for each action (0='right', 1='left', 2='wait'),")
    print(
        f"\tat the end of training, on a sample of mus extracted from the same Gaussian used for training:"
    )
    action_stats = episodes.copy()
    action_stats["rt"] *= train_agent_data["dt"]
    action_stats = action_stats.groupby("a").apply(f).reset_index()
    del f
    print(action_stats)

    # Generating a number of episodes with mu randomly extracted from
    # a finite set of values (for signal neutrality and scalar property)
    mus = (
        train_agent_data["sig"]
        * np.sqrt(train_agent_data["dt"] / shadlen_dt)
        * np.random.choice(
            mu_from_coherence([10 ** -3, 3.2, 6.4, 12.8, 25.6, 51.2]),
            6 * 10 ** 4,
        )
    )
    if "p_r_params" in train_agent_data:
        rewards = get_p_r(train_agent_data["p_r_params"])(mus)
    else:
        rewards = get_p_r(["one"])(mus)
    episodes, signals = generate_episodes(
        ws,
        taus=train_agent_data["taus"],
        dt=train_agent_data["dt"],
        t_max=train_agent_data["p_t_max_params"][1],
        sig=train_agent_data["sig"],
        sig_i=train_agent_data["sig_i"],
        gamma=train_agent_data["gamma"],
        mus=mus,
        rewards=rewards,
        use_t_integrators=train_agent_data["use_t_integrators"],
        p3_tot_abs=train_agent_data["p3_tot_abs"],
        collapse_on_abs_mu=False,
        return_signals=True,
        b_exp_i_taus=train_agent_data["b_exp_i_taus"],
    )

    # Computing the metrics for signal neutrality and scalar property
    (
        delta_sigma_diff_mean,
        delta_sigma_diff_max,
        delta_t,
    ) = compute_shadlen_dist(
        episodes,
        signals,
        ws,
        train_agent_data,
        max_delta_t=60,
        end_to_n_samples=100,
    )
    print(
        f"Signal neutrality dist = {delta_sigma_diff_mean:1.4f} (on {delta_t} time steps). This is the inverse of the signal neutrality measure in the paper."
    )

    g = episodes.copy()
    g = g[g["r"] > 0.0]
    g = g.groupby(["mu", "a"]).agg({"rt": ("mean", "std")}).reset_index()
    g[("rt", "cv")] = g[("rt", "std")] / g[("rt", "mean")]
    g = g.sort_values("mu")
    cv_mean = g[("rt", "cv")].mean()
    cv_max = g[("rt", "cv")].max()
    cv_min = g[("rt", "cv")].min()
    cv_mag_min_ratio = cv_max / cv_min
    del g

    print(f"Delta CV = {cv_max - cv_min:1.3f}. This is the inverse of the scalar property measure in the paper.")

    # Compute the curves for the plot à la Shadlen
    align2start, align2decision = compute_shadlen(
        episodes,
        signals,
        ws,
        dt=train_agent_data["dt"],
        n_t_max=None,
        p3_tot_abs=train_agent_data["p3_tot_abs"],
        start_to_percentile=50.0,
        end_to_n_samples=100,
    )

    print("This is a sample of the lines in a plot á la Shadlen:")
    print(align2decision.head())


if __name__ == "__main__":
    main()
