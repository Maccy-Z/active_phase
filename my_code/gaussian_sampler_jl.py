import numpy as np
from matplotlib import pyplot as plt
from utils import ObsHolder, make_grid, to_real_scale, points_within_radius
from config import Config
from jl_interface.GP_interface2 import JuliaGP

np.set_printoptions(precision=2)


# Sample phase diagrams from models.
def gen_pd(model: JuliaGP, xs, cfg, sample_cov=False) -> (np.ndarray, np.ndarray):
    """
    @param models: List of models for each phase
    @param xs: Points to sample from
    @param sample: Use mean or sample from GP.
    @return: Most likely phase diagram.
    """

    # Shape = [N_samples ** 2, N_phases]
    prob, fs = model.pred_proba_sampler(xs, full_cov=False, model_cov=sample_cov, nSamples=cfg.N_samples)
    prob_mean, prob_std = prob
    fs_mean, fs_std = fs

    pred_ys = np.argmax(prob_mean, axis=1)
    return pred_ys, prob_mean


def fit_gp(obs_holder: ObsHolder, cfg) -> JuliaGP:
    "Fit a new model to observations"
    X, y = obs_holder.get_obs()
    kern_var, kern_r = cfg.kern_var, cfg.kern_r
    model = JuliaGP()
    model.make_GP(X, y + 1, n_class=cfg.N_phases, init_sigma=kern_var, init_scale=kern_r, optimiser="Nothing")
    model.train_GP(n_iter=10)
    return model


def dist(pd1s: np.ndarray, pd2s: np.ndarray, weights):
    # pd.shape = [n_phases, N_samples ** 2]
    diffs = np.not_equal(pd1s, pd2s)
    mean_diffs = np.mean(diffs, axis=1)  # Mean over each phase diagram

    mean_diffs *= weights
    mean_diffs = np.sum(mean_diffs)

    return mean_diffs


# Predict new observaion and phase diagram
def gen_pd_new_point(old_model: JuliaGP, x_new, sample_xs, obs_probs, cfg):
    """Sample P_{n+1}(x_{n+1}, y), new phase diagrams assuming a new observation is taken at x_new.
    Returns: Phase diagrams and which phases were sampled"""

    X_old, y_old = old_model.X, old_model.y

    # Sample new models for each possible observation
    pds, sampled_phase = [], []
    for obs_phase, obs_prob in enumerate(obs_probs):
        # Ignore very unlikely observations
        if obs_prob < cfg.skip_phase:
            pds.append(np.empty((0, sample_xs.shape[0])))
            sampled_phase.append(0)
            continue

        y_new = obs_phase + 1
        X_new, y_new = np.vstack([X_old, x_new]), np.append(y_old, y_new)

        # Fit using old kernel parameters
        # TODO: Use individual kernel parameters for each phase instead of averaging
        kern_vars, kern_lens = np.mean(old_model.kern_var), np.mean(old_model.kern_lens)
        new_model = JuliaGP()
        new_model.make_GP(X_new, y_new, n_class=cfg.N_phases, init_sigma=kern_vars, init_scale=kern_lens, optimiser="Nothing")
        new_model.train_GP(n_iter=10)
        # Sample new phase diagrams, weighted to probability model is observed.
        prob, fs = new_model.pred_proba_sampler(sample_xs, full_cov=False, model_cov=False, nSamples=200)
        prob_mean, prob_std = prob

        pred_ys = np.argmax(prob_mean, axis=1)
        pds.append(pred_ys)
        sampled_phase.append(1)

    pds = np.stack(pds)  # Shape = [N_phases, N_samples ** 2]
    return pds, np.array(sampled_phase)


# Compute A(x) over all points
def acquisition(old_model: JuliaGP, new_Xs, cfg):
    # Grid over which to compute distances
    X_dist, _ = make_grid(cfg.N_dist, cfg.unit_extent)

    # P_n
    pd_old, _ = gen_pd(old_model, X_dist, sample_cov=False, cfg=cfg)  # For computing distances
    _, obs_probs = gen_pd(old_model, new_Xs, sample_cov=False, cfg=cfg)  # Probs for sampling A(x)

    # P_{n+1}
    avg_dists = []
    for new_X, obs_prob in zip(new_Xs, obs_probs):
        # print(max(obs_prob), obs_prob)
        # Skip point if certain enough
        if np.max(obs_prob) > cfg.skip_point:
            avg_dists.append(0)
            continue

        # Sample distance a region around selected point only
        mask = points_within_radius(X_dist, new_X, cfg.sample_dist, cfg.unit_extent)
        pd_old_want = pd_old[mask]
        X_dist_region = X_dist[mask]

        pd_new, sampled_phase = gen_pd_new_point(old_model, new_X, sample_xs=X_dist_region, obs_probs=obs_prob, cfg=cfg)

        # Expected distance between PD1 and PD2 averaged over all pairs
        pd_new = pd_new[sampled_phase]
        pd_old_want = np.tile(pd_old_want, (sum(sampled_phase), 1))
        weights = obs_prob[sampled_phase]
        avg_dist = dist(pd_old_want, pd_new, weights=weights)

        avg_dists.append(avg_dist)

    return np.array(avg_dists), obs_probs


# Suggest a single point to sample given current observations
def suggest_point(obs_holder, cfg: Config):
    # Fit to existing observations
    model = fit_gp(obs_holder, cfg=cfg)

    # Find max_x A(x)
    new_Xs, _ = make_grid(cfg.N_eval, cfg.unit_extent)  # Points to test for aquisition

    acq_fn, pd_probs = acquisition(model, new_Xs, cfg)
    max_pos = np.argmax(acq_fn)
    new_point = new_Xs[max_pos]
    prob_at_point = pd_probs[max_pos]

    # Rescale new point to real coordiantes
    new_point = to_real_scale(new_point, cfg.extent)

    # Plot old phase diagram
    X_display, _ = make_grid(cfg.N_display, cfg.unit_extent)
    pd_old_display = gen_pd(model, X_display, sample_cov=False, cfg=cfg)[0]

    return new_point, prob_at_point, (pd_old_display, acq_fn, pd_probs),


# Sample two points
def suggest_two_points(obs_holder, cfg):
    new_point, plot_probs, prob_at_point = suggest_point(obs_holder, cfg)
    obs_holder.make_obs(new_point)
    sec_point, _, _ = suggest_point(obs_holder, cfg)

    return new_point, sec_point, plot_probs
