import edit_source_files
import GPy
import numpy as np
import time
import scipy
from matplotlib import pyplot as plt

from utils import ObsHolder, make_grid, plot_scale, new_save_folder, array_scale
from config import Config

np.set_printoptions(precision=2)
cfg = Config()
N_phases = 3
save_dir = "./saves"

# Sample phase diagrams from models.
def gen_pd(models: list[GPy.core.GP], xs, sample=None):
    """
    @param models: List of models for each phase
    @param xs: Points to sample from
    @param sample: Use mean or sample from GP.
    @return: List of possible phase diagrams.
    """
    y_preds = []
    for phase_i, pd_model in enumerate(models):
        if sample is None:
            y_pred, _ = pd_model.predict(xs)  # m.shape = [n**2, 1]
        else:
            y_pred = pd_model.posterior_samples_f(xs, size=sample, method=cfg.normal_sample).squeeze(axis=1)
        y_preds.append(y_pred)

    y_preds = np.stack(y_preds)
    sample_pds = np.argmax(y_preds, axis=0).T
    return sample_pds


# Fit model to existing observations
def fit_gp(obs_holder: ObsHolder) -> list[GPy.core.GP]:
    "Trains a model for each phase."
    X, Y = obs_holder.get_obs()

    models = []
    for i in range(N_phases):
        phase_i = (Y == i) * 2 - 1  # Between -1 and 1

        kernel = GPy.kern.Matern52(input_dim=2, variance=4, lengthscale=1.)
        model = GPy.models.GPRegression(X, phase_i.reshape(-1, 1), kernel, noise_var=cfg.noise_var)
        #model.optimize(max_iters=1)

        models.append(model)
    return models


def dist2(pd1s: np.ndarray, pd2s: np.ndarray):
    #n_diagrams, n_points = pd1s.shape
    diffs = np.not_equal(pd1s, pd2s)
    mean_diffs = np.mean(diffs, axis=1)

    return np.mean(mean_diffs)


# Probability single gaussian is larger than the rest by integral of form f(x) exp(-x^2) where f(x) is product of CDFs.
def max_1(mu_1, sigma_1, mus, sigmas, hermite_roots):
    xs, weights = hermite_roots    # x to sample, weights for summation

    # Calculate CDF of each scaled gaussian
    cdfs = []
    for mu, sigma in zip(mus, sigmas, strict=True):
        scaled_xs = (np.sqrt(2) * sigma_1 * xs + mu_1 - mu) / sigma
        cdf_xs = scipy.stats.norm.cdf(scaled_xs)
        cdfs.append(cdf_xs)

    cdfs = np.stack(cdfs)

    # Product of cdfs for f(x)
    f = np.prod(cdfs, axis=0)

    prob = np.sum(weights * f) / np.sqrt(np.pi)

    return prob


# Probability each y is observed
def sample_new_y(models, x_new):
    """Probability of observing each phase at x_{n+1}. Everything is gaussian, so this is finding the max of gaussians."""
    mus, vars = [], []
    for model in models:
        mu, var = model.predict(x_new.reshape(-1, 2))
        mus.append(mu.squeeze()), vars.append(var.squeeze())

    sigmas = np.sqrt(vars).tolist()
    hm_roots = scipy.special.roots_hermite(30)

    probs = []
    for i, (mu, sigma) in enumerate(zip(mus, sigmas)):
        mu_remain = mus[:i] + mus[i + 1:]
        sigma_remain = sigmas[:i] + sigmas[i + 1:]

        probs.append(max_1(mu, sigma, mu_remain, sigma_remain, hm_roots))

    return np.array(probs)


# Predict new observaion and phase diagram
def gen_pd_new_point(models: list[GPy.core.GP], x_new, sample_xs, sample):
    """Sample P_{n+1}(x_{n+1}, y), new phase diagrams assuming a new observation is taken at x_new.
    Returns: Phase diagrams"""

    # First sample P_{n}(y_{n+1})
    pd_probs = sample_new_y(models, x_new)
    max_prob = max(pd_probs)

    # Skip sampling a point if certainty is already very high
    if max_prob > cfg.skip_point:
        return None, max_prob

    # Sample new models for each possible observation
    pds = []
    for obs_phase, obs_prob in enumerate(pd_probs):
        # Ignore very unlikely observations that will have 0 samples
        if obs_prob < cfg.skip_phase:
            # print(f'Prob observe phase {obs_phase}: {obs_prob:.3f}')
            pds.append(np.empty((0, sample_xs.shape[0])))
            continue

        new_models = []
        for phase_i, model in enumerate(models):
            kern_var, kern_len = float(model.kern.variance), float(model.kern.lengthscale)
            X, Y = model.X, model.Y

            y_new = 2 * int(phase_i == obs_phase) - 1
            X_new, Y_new = np.vstack([X, x_new, x_new]), np.vstack([Y, y_new, y_new])

            kernel = GPy.kern.Matern52(input_dim=2, variance=kern_var, lengthscale=kern_len)
            model = GPy.models.GPRegression(X_new, Y_new, kernel, noise_var=cfg.noise_var)

            if cfg.optim_step:
                model.optimize()

            new_models.append(model)

        # Sample new phase diagrams, weighted to probability model is observed
        if sample is None:
            count = round(1000 * obs_prob)
            pd_new = gen_pd(new_models, sample_xs, sample=None)
            pd_new = np.repeat(pd_new, count, axis=0)
        else:
            pd_new = gen_pd(new_models, sample_xs, sample= round(sample * obs_prob)) # Note, rounding here.

        pds.append(pd_new)


    pds = np.concatenate(pds)
    return pds, max_prob


def acquisition(models, new_Xs):
    # Grid over which to compute distances
    X_dist, X1_dist, X2_dist = make_grid(cfg.N_dist)

    # P_n
    pd_old = gen_pd(models, X_dist, sample=cfg.sample_old)

    # P_{n+1}
    avg_dists, max_probs = [], []
    for new_X in new_Xs:
        print(new_X, end=" ")

        if cfg.sample_dist is not None:
            # Sample distance a region around selected point only
            X_min = new_X - cfg.sample_dist / 2
            X_max = X_min + cfg.sample_dist

            X_min = np.maximum(X_min, cfg.xmin)
            X_max = np.minimum(X_max, cfg.xmax)

            # Create a mask of wanted points
            mask = (X_dist[:, 0] >= X_min[0]) & (X_dist[:, 0] <= X_max[0]) & (X_dist[:, 1] >= X_min[1]) & (X_dist[:, 1] <= X_max[1])

            pd_old_want = pd_old[:, mask]
            X_dist_region = X_dist[mask]

        else:
            X_dist_region = X_dist
            pd_old_want = pd_old

        pd_new, max_prob = gen_pd_new_point(models, new_X, sample_xs=X_dist_region, sample=cfg.sample_new)
        max_probs.append(max_prob)

        if pd_new is None:  # Skipped sampling point
            avg_dists.append(0)
            continue

        # Expected distance between PD1 and PD2 averaged over all pairs
        pd_old_repeat = np.repeat(pd_old_want, pd_new.shape[0], axis=0)
        pd_new_tile = np.tile(pd_new, (pd_old_want.shape[0], 1))
        avg_dist = dist2(pd_old_repeat, pd_new_tile)
        avg_dists.append(avg_dist)

    X_display, _, _ = make_grid(cfg.N_display, xmin=cfg.xmin, xmax=cfg.xmax)
    pd_old_mean = gen_pd(models, X_display, sample=None)[0]
    return pd_old_mean, np.array(avg_dists), np.array(max_probs)


def main():
    obs_holder = ObsHolder(Config())
    save_path = new_save_folder(save_dir)
    print()
    print(save_path)
    print()
    # Init observations to start off
    X_init, _, _ = make_grid(cfg.N_init)
    for xs in X_init:
        obs_holder.make_obs(xs)

    for i in range(30):
        st = time.time()

        # Fit to existing observations
        models = fit_gp(obs_holder)

        # Find max_x A(x)
        new_Xs, _, _ = make_grid(cfg.N_eval, xmin=cfg.xmin, xmax=cfg.xmax)  # Points to test for aquisition

        pd_old, avg_dists, max_probs = acquisition(models, new_Xs)
        max_pos = np.argmax(avg_dists)
        new_point = new_Xs[max_pos]
        obs_holder.make_obs(new_point)

        print()
        print()
        print(f'Iter: {i}, Time: {time.time() - st:.3g}s ' )

        print()
        print("New point to sample from:", new_point)

        # Plot probability of observaion
        plt.subplot(1, 3, 1)
        plt.title("P(obs) ")
        plt.imshow(1 - max_probs.reshape(cfg.N_eval, cfg.N_eval), extent=(-2, 2, -2, 2),
                       origin="lower", vmax=1, vmin=0)

        # Plot acquisition function
        plt.subplot(1, 3, 2)
        plt.title(f'Step {i}:  Acq. fn')

        sec_low = np.sort(np.unique(avg_dists))[1]
        try:
            plt.imshow(avg_dists.reshape(cfg.N_eval, cfg.N_eval), extent=(-2, 2, -2, 2),
                       origin="lower", vmin=sec_low)   # Phase diagram
        except ValueError:
            print(sec_low)
            pass

        # Plot current phase diagram and next sample point
        X_obs, Y_obs = obs_holder.get_obs()
        xs_train, ys_train = X_obs[:, 0], X_obs[:, 1]

        plt.subplot(1, 3, 3)
        plt.title(f"PD and points")

        plt.imshow(pd_old.reshape(cfg.N_display, cfg.N_display), extent=(-2, 2, -2, 2), origin="lower")   # Phase diagram
        plt.scatter(xs_train, ys_train, marker="x", s=30, c=Y_obs, cmap='bwr')   # Existing observations
        plt.scatter(new_point[0], new_point[1], s=80, c='tab:orange')      # New observations

        #plt.colorbar()

        plt.savefig(f'{save_path}/{i}.png', bbox_inches="tight")
        plt.show()

        obs_holder.save(save_path)


    obs_holder.plot_mean()
    # with open("./saves/both_sample", "wb") as f:
    #     pickle.dump(dists, f)




if __name__ == "__main__":

    main()


