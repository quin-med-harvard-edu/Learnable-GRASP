import numpy as np
from copy import copy


def soft_thresholding(x, lam):
    return (np.abs(x) - lam) * x / (np.abs(x) + np.power(.1, 6)) * (np.abs(x) > lam)


def singular_value_thresholding(x, lam):
    u, s, vt = np.linalg.svd(x, False)
    s_prime = soft_thresholding(s, lam)
    hat_x = np.dot(u * s_prime, vt)

    return hat_x, s_prime


def project_l2(x, c):
    return (np.abs(x) < c) * x + (np.abs(x) >= c) * c * x / np.abs(x)


def prox_tv(x, lam, max_iter):
    k = np.linalg.norm(x)
    p = np.zeros(x.shape - np.array([1, 0]))
    s = np.zeros(p.shape)
    tn = 1

    rh = None
    for idx_iter in range(max_iter):
        t = copy(tn)
        p_old = copy(p)

        tv_t_s = np.concatenate(
            [np.concatenate([[-1 * s[0]], s[:-1] - s[1:]]), [s[-1]]])
        tmp_ = project_l2(x - lam * tv_t_s, k)
        rh = s + (tmp_[1:] - tmp_[:-1]) / (8 * lam + np.power(.1, 15))
        rh /= np.maximum(np.abs(rh), 1)
        p = rh
        tn = (1 + np.sqrt(1 + np.power(t, 2) * 4)) / 2
        s = p + (t - 1) / tn * (p - p_old)

    tv_t_rh = np.concatenate(
        [np.concatenate([[-1 * rh[0]], rh[:-1] - rh[1:]]), [rh[-1]]])
    return project_l2(x - lam * tv_t_rh, k)


def prox_tv_l12(x, lam, max_iter):
    k = np.linalg.norm(x)
    p = np.zeros(x.shape - np.array([1, 0]))
    s = np.zeros(p.shape)
    tn = 1

    rh = None
    for idx_iter in range(max_iter):
        t = copy(tn)
        p_old = copy(p)

        tv_t_s = np.concatenate(
            [np.concatenate([[-1 * s[0]], s[:-1] - s[1:]]), [s[-1]]])
        tmp_ = project_l2(x - lam * tv_t_s, k)
        rh = s + (tmp_[1:] - tmp_[:-1]) / (8 * lam + np.power(.1, 15))
        rh /= np.maximum(np.abs(rh), 1)
        p = rh
        tn = (1 + np.sqrt(1 + np.power(t, 2) * 4)) / 2
        s = p + (t - 1) / tn * (p - p_old)

    tv_t_rh = np.concatenate(
        [np.concatenate([[-1 * rh[0]], rh[:-1] - rh[1:]]), [rh[-1]]])
    return project_l2(x - lam * tv_t_rh, k)
