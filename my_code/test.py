from jl_interface.GP_interface2 import JuliaGP
from utils import ObsHolder, make_grid
from config import Config
from matplotlib import pyplot as plt
import numpy as np
from gaussian_sampler_jl import suggest_point


def fit_obs(cfg):
    obs_holder = ObsHolder(cfg=cfg, save_path="/tmp/null")
    X_init = [[0, 0], [0, 1], [0, 2]]
    phase_init = [0, 1, 0]

    for phase, X in zip(phase_init, X_init, strict=True):
        obs_holder.make_obs(X, phase=phase)

    return obs_holder


def main():
    cfg = Config()
    obs_holder = fit_obs(cfg)

    new_point, prob_at_point, plot_pds = suggest_point(obs_holder, cfg)

    (pd_old_display, avg_dists, pd_probs) = plot_pds

    print(new_point, prob_at_point)

    pd_old_display = pd_old_display.reshape([cfg.N_display, cfg.N_display]).T   # Imshow uses (y, x)

    X_obs, phase_obs = obs_holder.get_og_obs()
    xs, ys = X_obs[:, 0], X_obs[:, 1]

    print(xs, ys)
    print(new_point)
    plt.imshow(pd_old_display, origin="lower", extent=(0, 2, 0, 2))
    plt.scatter(new_point[0], new_point[1], c="orange")
    plt.scatter(xs, ys, c = phase_obs, cmap="coolwarm")
    plt.show()


if __name__ == "__main__":
    main()
