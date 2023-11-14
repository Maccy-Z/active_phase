# jl = Julia(runtime='/home/maccyz/julia-1.9.3/bin/julia')
from julia import Main
import numpy as np

Main.include("./jl_interface/jl_GP_interface.jl")


class JuliaGP:
    X: np.ndarray
    y: np.ndarray
    kern_var: np.ndarray
    kern_lens: np.ndarray

    def __init__(self):
        self.sampler = None
        self.GP = None

    def make_sampler(self, n_gaussians):
        self.sampler = Main.make_sampler(n_gaussians)

    def make_GP(self, X, y, n_class: int, init_sigma: float, init_scale: float, optimiser: str):
        self.X = X
        self.y = y

        init_inv_scale = 1 / init_scale
        self.GP = Main.make_GP(X, y, n_class=n_class, init_sigma=init_sigma, init_scale=init_inv_scale, optimiser=optimiser)

    # Trains GP and returns the kernel params
    def train_GP(self, n_iter=100) -> tuple:
        kern_var, kern_inv_lens = Main.train_GP(self.GP, n_iter=n_iter)

        self.kern_var = kern_var
        self.kern_lens = 1 / np.array(kern_inv_lens)

        return kern_var, self.kern_lens

    # Predicts probabilities and std for test data and hidden state
    def pred_proba_sampler(self, X_test, full_cov: bool, model_cov: bool, nSamples: int) -> tuple[tuple, tuple]:
        if self.sampler is None:
            if model_cov:
                self.make_sampler(len(X_test) * nSamples)
            else:
                self.make_sampler(0)

        (probs, p_std), (f_mu, f_var) = Main.pred_proba_sampler(self.GP, X_test, full_cov=full_cov, model_cov=model_cov,
                                                                nSamples=nSamples, sampler=self.sampler)
        return (probs, p_std), (f_mu, f_var)

class JuliaGPHolder:
    def __init__(self):
        self.current_model = None
        self.models = []




