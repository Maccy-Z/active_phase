# Evaluate error of GP model
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from src.utils import make_grid, to_real_scale, skyrmion_pd
from src.config import Config

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from skimage import measure

Extent = ((0, 1.), (0., 1.), (0., 1.))


def acqusition_fn(obs, candidates):
    dists = []
    for point in candidates:
        # print(point)
        dist = np.linalg.norm(obs - point, axis=1)

        weighted_dist = np.exp(- 15 * dist)
        dists.append(np.mean(weighted_dist))
    return dists


# Vectorized boundary checking
def find_boundaries_vectorized(predictions):
    # Create boolean arrays for each shift in both directions along each axis
    shift_x_forward = predictions[:-1, :, :] != predictions[1:, :, :]
    shift_x_backward = predictions[1:, :, :] != predictions[:-1, :, :]

    shift_y_forward = predictions[:, :-1, :] != predictions[:, 1:, :]
    shift_y_backward = predictions[:, 1:, :] != predictions[:, :-1, :]

    shift_z_forward = predictions[:, :, :-1] != predictions[:, :, 1:]
    shift_z_backward = predictions[:, :, 1:] != predictions[:, :, :-1]

    # Combine shifts to identify boundaries
    bounds = np.zeros_like(predictions, dtype=bool)
    bounds[:-1, :, :] |= shift_x_forward
    bounds[1:, :, :] |= shift_x_backward

    bounds[:, :-1, :] |= shift_y_forward
    bounds[:, 1:, :] |= shift_y_backward

    bounds[:, :, :-1] |= shift_z_forward
    bounds[:, :, 1:] |= shift_z_backward

    return bounds


def suggest_point(Xs, labels):
    # Train the SVC
    clf = svm.SVC(decision_function_shape='ovr')
    clf.fit(Xs, labels)

    # Create a grid of points
    grid_points, mesh = make_grid(11, Extent)

    # Get the decision function for each point on the grid
    Z = clf.decision_function(grid_points)

    # Get the class with the highest decision function for each point
    Z = Z.reshape(11, 11, 11, -1)

    y_pred = np.argmax(Z, axis=-1).reshape(11, 11, 11)

    boundaries = find_boundaries_vectorized(y_pred)

    grid_points = grid_points.reshape(11, 11, 11, 3)
    boundary_points = grid_points[boundaries]

    # print(boundary_points)

    dists = acqusition_fn(Xs, boundary_points)
    min_idx = np.argmin(dists)
    min_point = boundary_points[min_idx]

    return min_point, None, y_pred


def plot(Xs, labels, new_point, contours, pred_pd):
    pred_pd = pred_pd[:, :, 5].T
    mask = (Xs[:, 2] > 0.4) & (Xs[:, 2] < 0.6)
    Xs = Xs[mask]
    labels = np.array(labels)[mask]

    # Plot the results
    plt.figure(figsize=(3, 3))

    plt.scatter(Xs[:, 0], Xs[:, 1], marker="x", s=30, c=labels, cmap='bwr')  # , c=labels, cmap='viridis')
    plt.imshow(pred_pd, extent=(0, 1, 0, 1), origin="lower")  # Phase diagram
    if new_point is not None:
        plt.scatter(new_point[0], new_point[1], c='r')
    if contours is not None:
        for contor in contours:
            plt.plot(*zip(*contor), c='k', linestyle='dashed')

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    # plt.yticks(np.linspace(-2, 2, 5))
    # plt.xticks(np.linspace(-2, 2, 5))
    plt.subplots_adjust(left=0.1, right=0.99, top=0.925, bottom=0.1, wspace=0.4, hspace=0.4)

    plt.show()


def error(pred_pd, pd_fn):
    points, _ = make_grid(pred_pd.shape[0], Extent)
    points = to_real_scale(points, Extent)
    true_pd = []
    for X in points:
        true_phase = pd_fn(X, train=False)
        true_pd.append(true_phase)

    true_pd = np.stack(true_pd).reshape(pred_pd.shape)

    diff = np.not_equal(pred_pd, true_pd)
    diff_mean = np.mean(diff)
    #
    # print(points)
    # plot(np.array([[0, 0]]), np.array([0]), None, None, true_pd)
    return diff_mean


def main():
    pd_fn = skyrmion_pd

    Xs = [[0.3, 0.4, 0.5, 0.6, 0, 0.7], [0, 0.1, 0, 0.5, 0, 0.8], [0, 0.1, 0.5, 0.3, 0.4, 1]]
    Xs = np.array(Xs).T
    # Xs = np.array([[0, 0, 0], [0, .1, .1], [1, 0.2, 0.8]])
    # Xs, _ = make_grid(11, Extent)

    labels = [pd_fn(X, train=False) for X in Xs]
    print(labels)
    errors = []
    for i in range(251):
        new_point, contors, full_pd = suggest_point(Xs, labels)

        Xs = np.append(Xs, [new_point], axis=0)
        labels.append(pd_fn(new_point))

        pred_error = error(full_pd, pd_fn)
        errors.append(pred_error)
        if i % 50 == 0:
            # plot(Xs, labels, new_point, None, full_pd)
            print(pred_error)

    plot(Xs, labels, None, None, full_pd)

    # plt.imshow(full_pd, origin="lower")
    # plt.show()

    print(errors)
    print(len(errors))


if __name__ == "__main__":
    main()