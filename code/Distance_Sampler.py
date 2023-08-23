from gaussian_sampler import suggest_point, suggest_two_points
from matplotlib import pyplot as plt
from matplotlib import axes as Axes
import numpy as np

from utils import ObsHolder, make_grid, new_save_folder, tri_pd
from config import Config


class DistanceSampler:
    def __init__(self, init_phases, init_Xs, cfg, save_dir="./saves"):
        save_path = new_save_folder(save_dir)
        print(f'{save_path = }')
        print()

        self.cfg = cfg
        self.obs_holder = ObsHolder(self.cfg, save_path=save_path)

        # Check if initial inputs are within allowed area
        vectors = np.array(init_Xs)
        xmin, xmax, ymin, ymax = self.cfg.extent
        inside_x = np.logical_and(vectors[:, 0] >= xmin, vectors[:, 0] <= xmax)
        inside_y = np.logical_and(vectors[:, 1] >= ymin, vectors[:, 1] <= ymax)
        in_area = np.all(inside_x & inside_y)
        if not in_area:
            print("\033[31mWarning: Observations are outside search area\033[0m")

        # Check phases are allowed

        for phase, X in zip(init_phases, init_Xs, strict=True):
            self.obs_holder.make_obs(X, phase=phase)

    # Load in axes for plotting
    def set_plots(self, axs):
        self.p_obs_ax: Axes = axs[0]
        self.acq_ax: Axes = axs[1]
        self.pd_ax: Axes = axs[2]

    def plot(self, first_point, pd_old, avg_dists, pd_probs, sec_point=None):
        self.p_obs_ax.clear()
        self.acq_ax.clear()
        self.pd_ax.clear()

        # Reshape 1d array to 2d
        side_length = int(np.sqrt(len(pd_old)))
        pd_old = pd_old.reshape((side_length, side_length))
        side_length = int(np.sqrt(len(avg_dists)))
        avg_dists = avg_dists.reshape((side_length, side_length))
        max_probs = np.amax(pd_probs, axis=1)
        side_length = int(np.sqrt(len(max_probs)))
        max_probs = max_probs.reshape((side_length, side_length))

        # Plot probability of observaion
        self.p_obs_ax.set_title("P(obs)")
        self.p_obs_ax.imshow(1 - max_probs, extent=self.cfg.extent,
                             origin="lower", vmax=1, vmin=0)

        # Plot acquisition function
        self.acq_ax.set_title(f'Acq. fn')
        sec_low = np.sort(np.unique(avg_dists))[1]
        self.acq_ax.imshow(avg_dists, extent=self.cfg.extent,
                           origin="lower", vmin=sec_low)  # Phase diagram

        # Plot current phase diagram and next sample point
        X_obs, phase_obs = self.obs_holder.get_obs()
        xs_train, ys_train = X_obs[:, 0], X_obs[:, 1]
        self.pd_ax.set_title(f"PD and points")
        self.pd_ax.imshow(pd_old, extent=self.cfg.extent, origin="lower")  # Phase diagram
        self.pd_ax.scatter(xs_train, ys_train, marker="x", s=30, c=phase_obs, cmap='bwr')  # Existing observations
        self.pd_ax.scatter(first_point[0], first_point[1], s=80, c='tab:orange')  # New observations
        if sec_point is not None:
            self.pd_ax.scatter(sec_point[0], sec_point[1], s=80, c='r')  # New observations

        # plt.colorbar()

    def make_obs(self, phase: int, X: np.ndarray):
        self.obs_holder.make_obs(X, phase=phase)
        self.obs_holder.save()

        new_point, (pd_old, avg_dists, pd_probs), prob_at_point = suggest_point(self.obs_holder, self.cfg)
        self.plot(new_point, pd_old, avg_dists, pd_probs)

        return new_point, prob_at_point

    def first_obs(self):
        new_point, (pd_old, avg_dists, pd_probs), prob_at_point = suggest_point(self.obs_holder, self.cfg)
        self.plot(new_point, pd_old, avg_dists, pd_probs)

        return new_point, prob_at_point

    def dual_obs(self):
        first_point, sec_point, (pd_old, avg_dists, pd_probs) = suggest_two_points(self.obs_holder, self.cfg)
        self.plot(first_point, pd_old, avg_dists, pd_probs, sec_point=sec_point)

        return first_point, sec_point




def main(save_dir):
    print(save_dir)
    cfg = Config()

    # Init observations to start off
    X_init, _, _ = make_grid(cfg.N_init, cfg.extent)
    phase_init = [None for _ in X_init]

    distance_sampler = DistanceSampler(phase_init, X_init, cfg, save_dir=save_dir)

    fig = plt.figure(figsize=(10, 3.3))

    # Create three subplots
    axes1 = fig.add_subplot(131)
    axes2 = fig.add_subplot(132)
    axes3 = fig.add_subplot(133)

    distance_sampler.set_plots([axes1, axes2, axes3])

    for _ in range(cfg.steps):
        new_points = distance_sampler.dual_obs()
        fig.show()

        for p in new_points:
            obs_phase = tri_pd(p)
            distance_sampler.obs_holder.make_obs(p, obs_phase)



if __name__ == "__main__":
    save_dir = "./saves"

    main(save_dir)
