# Evaluate error of GP model
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

cwd = os.getcwd()
os.chdir(os.path.dirname(cwd))

import pickle
import numpy as np
import math
from matplotlib import pyplot as plt
import copy

from my_code.utils import make_grid, ObsHolder, c_print
from my_code.config import Config
from my_code.gaussian_sampler import fit_gp, gen_pd


# Distance between true PD and prediction
def dist(obs_holder, *, true_pd, cfg, points, t):
    # Truncate observations up to time t, assuming first 2 observations are given.
    Xs, Ys = obs_holder.get_og_obs()
    T = t + 2
    obs_holder._obs_pos = Xs[:T]
    obs_holder.obs_phase = Ys[:T]

    model = fit_gp(True, obs_holder, cfg=cfg)

    X_eval, _ = make_grid(points, cfg.unit_extent)
    probs = gen_pd(model, X_eval, cfg=cfg)
    pd_pred = np.argmax(probs, axis=1).reshape((points,) * cfg.N_dim)
    # pd_pred = np.flip(np.rot90(np.rot90(pd_pred)), axis=1)

    true_pd = np.stack(true_pd).reshape((points,) * cfg.N_dim)

    diff = np.not_equal(pd_pred, true_pd)
    diff_mean = np.mean(diff)

    print("\033[91mDelete this to stop plotting\033[0m")
    pd_pred[10, 0, 0] = 2
    plt.imshow(pd_pred[:, :, 0], origin='lower')
    plt.show()

    return diff_mean


def deduplicate(array_list):
    seen = []  # List to store the distinct arrays seen so far
    count_dict = {}  # Dictionary to store the result

    count = 0
    for idx, arr in enumerate(array_list):
        # Check if the current array is identical to any of the arrays seen so far
        if not any(np.array_equal(arr, seen_arr) for seen_arr in seen):
            seen.append(arr)
            count += 1
        count_dict[idx] = count

    reversed_dict = {}
    for key, value in count_dict.items():
        if value not in reversed_dict:
            reversed_dict[value] = key
    return reversed_dict


def main():
    true_pd = np.zeros((21, 21, 21))
    true_pd[10:] = 1

    # np.array([[2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
    #                     [1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
    #                     [1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
    #                     [1., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
    #                     [1., 1., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
    #                     [1., 1., 1., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
    #                     [1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
    #                     [1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
    #                     [1., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
    #                     [1., 1., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
    #                     [1., 1., 1., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
    #                     [1., 1., 1., 1., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
    #                     [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
    #                     [0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
    #                     [0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 2., 2., 2.],
    #                     [0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2.],
    #                     [0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 2.],
    #                     [0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
    #                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
    #                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1.],
    #                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    eval_points = true_pd.shape[0]
    c_print(f'Number of evaluation points = {eval_points}', color='yellow')
    c_print(f'Note: Plotting is done with x and y axis inverted at z=0', color='yellow')
    assert len(true_pd) != 0, ("Fill in the true phase diagram as a 2D numpy array, with the same number of points as eval_points. "
                               "It might need transposing / reflecting to orient properly. ")
    plt.imshow(true_pd[:, :, 0], origin='lower')
    plt.show()

    f = sorted([int(s) for s in os.listdir("./saves")])
    save_name = f[-1]
    print(f'{save_name = }')

    og_obs = ObsHolder.load(f'./saves/{save_name}')
    # Deduplicate observations
    all_obs = og_obs._obs_pos
    obs_map = deduplicate(all_obs)

    with open(f'./saves/{save_name}/cfg.pkl', "rb") as f:
        cfg: Config = pickle.load(f)
    assert true_pd.ndim == cfg.N_dim, "Observation dimension must be the same as model dimension"

    errors = []
    for t in range(len(og_obs.obs_phase)):
        obs_holder = copy.deepcopy(og_obs)
        error = dist(obs_holder, true_pd=true_pd, points=eval_points, t=t, cfg=cfg)
        errors.append(error)
    plt.plot(errors)
    plt.show()
    #
    # print()
    # for s in [10, 20, 30, 40, 50]:
    #     print(f'{errors[s]}')

    print("Errors:")
    print("Copy me to error_plot.py")
    print(errors)
    print("Map number of observations made to number of unique observations (remove duplicates)")
    print("Copy me to error_plot.py")
    print(obs_map)


if __name__ == "__main__":
    main()
