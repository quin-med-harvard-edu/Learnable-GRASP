import numpy as np
from copy import copy
from tqdm import tqdm

from rec_algo.lowrank_sparse.helpers import soft_thresholding, \
    singular_value_thresholding, prox_tv


class LowRankSparseTV(object):

    def __init__(self, k_samples: np.ndarray,
                 sqrt_dcf: np.ndarray,
                 coil_p: np.ndarray,
                 nufft_obj: object,
                 lam_low_rank: float = 20,
                 lam_sparse: float = 0.250):
        if len(k_samples.shape) == 4:
            assert k_samples.shape[0] == sqrt_dcf.shape[0], \
                "LowRankSparse: number of vol mismatch between input variables!"

        self.k_samples = k_samples
        self.sqrt_dcf = sqrt_dcf
        self.coil_p = coil_p
        self.lam_low_rank = lam_low_rank
        self.lam_sparse = lam_sparse

        self.nufft_obj = nufft_obj

    def reconstruct(self,
                    k_data: np.ndarray,
                    s_init: np.ndarray,
                    l_init: np.ndarray,
                    max_num_iter: int = 20,
                    tk: float = 0.05,
                    tol_level: float = 0.0025):

        hat_m = self.nufft_obj.adjoint(k_data, self.k_samples, self.sqrt_dcf,
                                       self.coil_p)
        hat_m = np.sum(hat_m, axis=1)

        vec_m = np.array([hat_m[idx_, :, :].flatten() for idx_ in range(hat_m.shape[0])])
        k_dom = self.nufft_obj.forward(hat_m, self.k_samples, self.sqrt_dcf, self.coil_p)
        scale_ = np.max(np.abs(k_dom)) / np.max(np.abs(k_data))

        hat_l = copy(l_init)
        hat_s = copy(s_init)

        vec_l = np.array([hat_l[idx_, :, :].flatten() for idx_ in range(hat_m.shape[0])])
        vec_s = np.array([hat_s[idx_, :, :].flatten() for idx_ in range(hat_m.shape[0])])

        pbar = tqdm(range(max_num_iter))

        list_error = []
        for idx_iter in pbar:
            vec_m_old = copy(vec_m)

            tmp_ = np.array(
                [np.reshape((vec_s + vec_l)[idx_, :], hat_m[idx_, :, :].shape) for idx_
                 in range(hat_m.shape[0])])
            hat_k = self.nufft_obj.forward(copy(tmp_), self.k_samples, self.sqrt_dcf,
                                           self.coil_p)
            residual = hat_k - k_data * scale_

            delta = self.nufft_obj.adjoint(residual, self.k_samples, self.sqrt_dcf,
                                           self.coil_p)
            delta = np.sum(delta, axis=1)

            vec_delta = np.array(
                [delta[idx_, :, :].flatten() for idx_ in range(hat_m.shape[0])])

            vec_l = vec_l - vec_delta * tk
            vec_s = vec_s - vec_delta * tk

            vec_l, s_prime = singular_value_thresholding(vec_l, tk * self.lam_low_rank)
            vec_s = prox_tv(vec_s, tk * self.lam_sparse, 10)

            vec_m = vec_l + vec_s
            err_ = (np.linalg.norm(residual) ** 2) + self.lam_low_rank * np.sum(
                s_prime) + self.lam_sparse * np.sum(np.abs(vec_s[1:] - vec_s[:-1]))
            list_error.append(err_)
            pbar.set_description("obj:{}, dif:{}".format(
                np.around(err_, 3),
                np.around((np.linalg.norm(np.abs(vec_m - vec_m_old)) / np.linalg.norm(
                    np.abs(vec_m_old))), 5)))
            if (np.linalg.norm(vec_m_old - vec_m) / np.linalg.norm(
                    vec_m_old)) < tol_level:
                break

        hat_s = np.array(
            [np.reshape(vec_s[idx_, :], hat_m[idx_, :, :].shape) for idx_ in
             range(hat_m.shape[0])])
        hat_l = np.array(
            [np.reshape(vec_l[idx_, :], hat_m[idx_, :, :].shape) for idx_ in
             range(hat_m.shape[0])])

        return hat_l, hat_s, np.array(list_error)
