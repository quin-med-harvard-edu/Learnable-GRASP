import numpy as np
from copy import copy
from tqdm import tqdm
import SimpleITK as sitk

eps = np.power(.1, 15)  # Serves well inside log and as an addition to a denominator


class MotionGRASP:
    """ GRASP algorithm for MRI image reconstruction
        Solves min_x ||Ax-y||_2 + lam ||TV(x)||_1 with conjugate gradient descent method
        Attr:
            k_samples(ndarray[float]): [Nv x NS x NSp] star sampling information
            sqrt_dcf(ndarray[complex]): [Nv x NS x NSp] square root of the dcf matrix
            coil_p(ndarray[complex]): [NS x NS x NC] coil profile matrix
                [Nv,NS,NC,NSp] = [num_volume,num_sample, num_coil, num_spoke]
            nufft_obj(object(NUFFT)): check "./helpers/nufft_lib"
            lam(float): linear combination parameter for the optimization
            register_obj(object(SimpleElastix)): SimpleElastixModule for registration
        Methods:
            reconstruct: Regularized image reconstruction procedure.
            _objective: Computes ||Ax-y||_2 + lam ||TV(x)||_1 for a given x
            _derivative: Computes del(||Ax-y||_2 + lam ||TV(x)||_1)/del(x) for x
                    """

    def __init__(self,
                 k_samples: np.ndarray,
                 sqrt_dcf: np.ndarray,
                 coil_p: np.ndarray,
                 nufft_obj: object,
                 lam: float = None,
                 register_obj=object):

        if len(k_samples.shape) == 4:
            assert k_samples.shape[0] == sqrt_dcf.shape[0], \
                "GRASP: number of vol mismatch between input variables!"

        self.k_samples = k_samples
        self.sqrt_dcf = sqrt_dcf
        self.coil_p = coil_p
        self.lam = lam

        self.nufft_obj = nufft_obj
        self.register_obj = register_obj

    def reconstruct(self,
                    x: np.ndarray,
                    Y: np.ndarray,
                    max_num_iter: int = 20,
                    min_num_iter: int = 9,
                    max_num_line_search: int = 150,
                    alpha: float = 0.01,
                    beta: float = 0.6,
                    tol_grad: float = np.power(.1, 3)):
        """ Given initial x and observation y, solves ||Ax-Y||_2 + lam ||TV(x)||_1
            with conjugate gradient descent (CGD) with a nested line search (LS)
            algorithm. Terminates CGD if maximum number of iterations is reached or
            the norm of the gradient is below a tolerance.
            Args:
                x(ndarray[complex]): [Nv x NS x NS] image domain representation
                Y(ndarray[complex]): [Nv x NS x NC x NSp] k_space observation
                    [NS,NC,NSp,Nv] = [num_sample, num_coil, num_spoke, num_volume]
                max_num_iter(int): maximum number CGD iterations
                min_num_iter(int): minimum number of CGD iterations
                max_num_line_search(int): maximum number of LS steps
                alpha(float): line search gradient scale for LS termination
                beta(float): line search scale
                tol_grad(float): gradient tolerance for CGD termination
            Return:
                x_rec(ndarray[complex]): [NS x NS x Nv] reconstructed image
                """
        # print("Grasp Reconstruction")  # Report user what's going on

        # TODO: This is not the smartest way to assign an attr
        if self.lam is None:
            self.lam = 0.0125 * np.max(np.abs(x))

        x_rec = copy(x)

        normy = np.sum(np.abs(Y) ** 2)

        # y = i-nufft(Y)
        # print("i-NUFFT y")
        y = self.nufft_obj.adjoint(Y, batch_k_samples=copy(self.k_samples),
                                   batch_sqrt_dcf=copy(self.sqrt_dcf),
                                   coil_p=copy(self.coil_p))

        # X = nufft(x)
        # print("NUFFT x")
        recX = self.nufft_obj.forward(x_rec, batch_k_samples=copy(self.k_samples),
                                      batch_sqrt_dcf=copy(self.sqrt_dcf),
                                      coil_p=copy(self.coil_p))

        # xX = i-nufft(nufft(x)) != x be careful about this property of the operator!
        # print("i-NUFFT(NUFFT) x")
        recxX = self.nufft_obj.adjoint(recX, batch_k_samples=copy(self.k_samples),
                                       batch_sqrt_dcf=copy(self.sqrt_dcf),
                                       coil_p=copy(self.coil_p))

        g0 = self._derivative_pre_comp(x_rec, recxX, y)

        dx = -g0
        # print("NUFFT 1st derivative!")
        dX = self.nufft_obj.forward(dx, batch_k_samples=copy(self.k_samples),
                                    batch_sqrt_dcf=copy(self.sqrt_dcf),
                                    coil_p=copy(self.coil_p))

        # print("i-NUFFT(NUFFT) 1st derivative!")
        dxX = self.nufft_obj.adjoint(dX, batch_k_samples=copy(self.k_samples),
                                     batch_sqrt_dcf=copy(self.sqrt_dcf),
                                     coil_p=copy(self.coil_p))

        reg_map, reg_i_map = None, None
        t_ = 1  # gradient scaling factor
        # Conjugate Gradient Descent
        tqdm_bar = tqdm(range(max_num_iter), desc='Grasp rec:', leave=True)
        for iter_rec in tqdm_bar:
            for iter_grasp in range(4):
                f_d = self._objective_pre_comp(x_rec, recX, Y, reg_map)
                # Line Search
                scale_grad = beta
                for iter_line_search in range(max_num_line_search):
                    scale_grad = t_ * np.power(beta, iter_line_search)
                    f_r = self._objective_pre_comp(x_rec + scale_grad * dx,
                                                   recX + scale_grad * dX, Y, reg_map)
                    if f_r <= (f_d - alpha * scale_grad * np.sum(np.abs(dx * g0))):
                        break

                if iter_line_search > 2:
                    t_ *= beta
                elif iter_line_search == 0:
                    t_ /= beta

                # Take the gradient step
                x_rec = x_rec + scale_grad * dx
                recX = recX + scale_grad * dX
                recxX = recxX + scale_grad * dxX

                # conjugate gradient calculation
                g1 = self._derivative_pre_comp(x_rec, recxX, y, reg_map, reg_i_map)
                beta_cg = np.sum(np.abs(g1) ** 2) / (np.sum(np.abs(g0) ** 2) + eps)
                g0 = copy(g1)

                dx = - g1 + beta_cg * dx
                dX = self.nufft_obj.forward(dx, batch_k_samples=copy(self.k_samples),
                                            batch_sqrt_dcf=copy(self.sqrt_dcf),
                                            coil_p=copy(self.coil_p))

                dxX = self.nufft_obj.adjoint(dX, batch_k_samples=copy(self.k_samples),
                                             batch_sqrt_dcf=copy(self.sqrt_dcf),
                                             coil_p=copy(self.coil_p))

                # Check if gradient norm is below tolerance
                if (iter_rec >= min_num_iter) and \
                        (np.linalg.norm(dx) < tol_grad):
                    break

            if (iter_rec) == 1:
                reg_map, reg_i_map = self._compute_registration_map(x_rec)
            tqdm_bar.set_description("cost:{} grad:{}"
                                     .format(f_r / normy, np.linalg.norm(dx) / normy))

        return self._register_with_map(x_rec, reg_map)

    def _objective_pre_comp(self, x, X, Y, reg_map=None):
        """ The objective utilizes NUFFT which is a linear operator. It can be useful
            to precompute some of the inverses and forward maps!
            Calculates the objective of ||Ax-Y||_2 + lam ||TV(x)||_1
            Observe that the derivative will include X = Ax that is precomputed.
            Args:
                x(ndarray[complex]): [Nv x NS x NS] image domain representation
                X(ndarray[complex]): [Nv x NS x NC x NSp] k_space observation with pre
                    computed forward operation
                Y(ndarray[complex]): [Nv x NS x NC x NSp] k_space observation
                    [NS,Nv] = [num_sample, num_volume]
            Return:
                derivative_value(ndarray[complex]): [Nv x NS x NS] derivative """
        multi_flag = (len(x.shape) > 2)
        if multi_flag:
            # Calculate ||X-y||_2
            tmp_ = X - Y
            l2_obj = np.sum(np.abs(tmp_) ** 2)

            # Calculate ||TV(x)||_1
            if self.lam:
                if reg_map:
                    x = self._register_with_map(x, reg_map)
                tmp_ = x[1:, :, :] - x[:-1, :, :]
                l1_obj = np.sum(np.abs(tmp_))
            else:
                l1_obj = 0
        else:
            raise (NotImplementedError)

        return l2_obj + self.lam * l1_obj

    def _derivative_pre_comp(self, x, xX, y, reg_map=None, reg_i_map=None):
        """ Calculates the derivative of ||Ax-Y||_2 + lam ||TV(x)||_1
            Observe that the derivative will include xX = A'Ax and y = A'Y and these are
            already pre computed and passed to the method.
            Args:
                x(ndarray[complex]): [Nv x NS x NS] image domain representation
                xX(ndarray[complex]): [Nv x NS x NS] image domain representation with
                    pre computed forward backward operation
                y(ndarray[complex]): [Nv x NS x NS] image domain representation
                    [NS,Nv] = [num_sample, num_volume]
            Return:
                derivative_value(ndarray[complex]): [Nv x NS x NS] derivative """

        multi_flag = (len(x.shape) > 2)
        if multi_flag:
            # Calculate del(||Ax-y||_2)/d(x)|_x
            del_l2 = 2 * np.array(xX - y)

            # Calculate del(||TV(x)||_1)/d(x)|_x
            if self.lam:
                if reg_map:
                    x = self._register_with_map(x, list_map=reg_map)
                tmp_ = x[1:, :, :] - x[:-1, :, :]
                tmp_ = tmp_ / (np.abs(tmp_) + eps)
                del_l1 = np.concatenate(([-tmp_[0, :, :]],
                                         tmp_[:-1, :, :] - tmp_[1:, :, :],
                                         [tmp_[-1, :, :]]), axis=0)
                if reg_map:
                    del_l1 = self._register_with_map(del_l1, list_map=reg_i_map)
            else:
                del_l1 = 0
        else:
            raise (NotImplementedError)

        return del_l2 + self.lam * del_l1

    def _compute_registration_map(self, xk_hat):

        _xk = copy(xk_hat)
        list_map = [None for idx in range(_xk.shape[0])]
        list_i_map = [None for idx in range(_xk.shape[0])]
        for idx_ in range(_xk.shape[0]):
            if idx_ == int(_xk.shape[0] / 2):
                pass
            else:
                self.register_obj.SetFixedImage(sitk.GetImageFromArray(
                    np.abs(_xk[int(_xk.shape[0] / 2), :, :])))
                self.register_obj.SetMovingImage(sitk.GetImageFromArray(
                    np.abs(_xk[idx_, :, :])))
                self.register_obj.Execute()
                list_map[idx_] = self.register_obj.GetTransformParameterMap()

                self.register_obj.SetFixedImage(sitk.GetImageFromArray(
                    np.abs(_xk[int(_xk.shape[0] / 2), :, :])))
                self.register_obj.SetMovingImage(self.register_obj.GetResultImage())
                self.register_obj.Execute()
                list_i_map[idx_] = self.register_obj.GetTransformParameterMap()
                # TODO: Iteratively update registration map

        return list_map, list_i_map

    def _register_with_map(self, xk_hat, list_map):
        _xk = copy(xk_hat)
        for idx in range(len(list_map)):
            if list_map[idx] != None:
                reg_x_real = sitk.GetArrayFromImage(
                    sitk.Transformix(sitk.GetImageFromArray(np.real(_xk[idx, :, :])),
                                     list_map[idx]))
                reg_x_imag = sitk.GetArrayFromImage(
                    sitk.Transformix(sitk.GetImageFromArray(np.imag(_xk[idx, :, :])),
                                     list_map[idx]))
                _xk[idx, :, :] = reg_x_real + 1j * reg_x_imag

        return _xk
