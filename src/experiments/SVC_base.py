# Evaluate error of GP model
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from src.utils import make_grid, bin_pd, quad_pd, tri_pd, to_real_scale
from src.config import Config

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from skimage import measure


# Generate some example data


def acqusition_fn(obs, candidates):
    dists = []
    for point in candidates:
        # print(point)
        dist = np.linalg.norm(obs - point, axis=1)

        weighted_dist = np.exp(- 2 * dist)
        dists.append(np.mean(weighted_dist))
    return dists


def suggest_point(Xs, labels):
    # Train the SVC
    clf = svm.SVC(decision_function_shape='ovr')
    clf.fit(Xs, labels)

    # Create a grid of points
    x = np.linspace(-2, 2, 19)
    y = np.linspace(-2, 2, 19)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Get the decision function for each point on the grid
    Z = clf.decision_function(grid_points)
    if Z.ndim > 1:
        # Get the class with the highest decision function for each point
        Z = Z.reshape(*xx.shape, -1)

        y_pred = np.argmax(Z, axis=2)

        contour1 = measure.find_contours(y_pred, 0.5)
        contour2 = measure.find_contours(y_pred, 1.5)
        contour3 = measure.find_contours(y_pred, 2.5)

        # The contours are given in (row, column) coordinates, so you need to swap the axes and rescale them to get the (x, y) coordinates
        contour1 = [(y / y_pred.shape[1] * 4 - 2, x / y_pred.shape[0] * 4 - 2) for x, y in contour1[0]]
        contour2 = [(y / y_pred.shape[1] * 4 - 2, x / y_pred.shape[0] * 4 - 2) for x, y in contour2[0]]
        try:
            contour3 = [(y / y_pred.shape[1] * 4 - 2, x / y_pred.shape[0] * 4 - 2) for x, y in contour3[0]]
            contors = np.concatenate([contour3, contour2, contour1])

        except IndexError as e:
            contors = np.concatenate([contour2, contour1])

        dists = acqusition_fn(Xs, contors)
        min_point = np.argmin(dists)

        preds = Z.reshape(*xx.shape, -1)
        preds = np.argmax(preds, axis=2)

        return contors[min_point], [contour1, contour2, contour3], preds

    else:
        Z = Z.reshape(xx.shape)

        contour = np.array(measure.find_contours(Z, 0.))
        contour = [(y / Z.shape[1] * 4 - 2, x / Z.shape[0] * 4 - 2) for x, y in contour[0]]

        dists = acqusition_fn(Xs, contour)
        min_point = np.argmin(dists)

        return contour[min_point], [contour], Z > 0


def plot(Xs, labels, new_point, contours, pred_pd):
    # Plot the results
    plt.figure(figsize=(3, 3))
    plt.title("4-phase diagram")

    plt.scatter(Xs[:, 0], Xs[:, 1], marker="x", s=30, c=labels, cmap='bwr')  # , c=labels, cmap='viridis')
    plt.imshow(pred_pd, extent=(-2, 2, -2, 2), origin="lower")  # Phase diagram
    if new_point is not None:
        plt.scatter(new_point[0], new_point[1], c='r')
    if contours is not None:
        for contor in contours:
            plt.plot(*zip(*contor), c='k', linestyle='dashed')

    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.yticks(np.linspace(-2, 2, 5))
    plt.xticks(np.linspace(-2, 2, 5))
    plt.subplots_adjust(left=0.1, right=0.99, top=0.925, bottom=0.1, wspace=0.4, hspace=0.4)
    #plt.tight_layout()
    plt.show()


def error(pred_pd, pd_fn):
    points, _ = make_grid(pred_pd.shape[0], Config().extent)
    points = to_real_scale(points, ((-2, 2), (-2, 2)))
    true_pd = []
    for X in points:
        true_phase = pd_fn(X, train=False)[0]
        true_pd.append(true_phase)

    true_pd = np.stack(true_pd).reshape(pred_pd.shape).T

    diff = np.not_equal(pred_pd, true_pd)
    diff_mean = np.mean(diff)
    #
    # print(points)
    # plot(np.array([[0, 0]]), np.array([0]), None, None, true_pd)
    return diff_mean


def main():
    pd_fn = quad_pd

    Xs = np.array([[0, -1.25], [0, 1]])

    labels = [pd_fn(X, train=False)[0] for X in Xs]

    errors = []
    for i in range(101):
        new_point, contors, full_pd = suggest_point(Xs, labels)

        Xs = np.append(Xs, [new_point], axis=0)
        labels.append(pd_fn(new_point)[0])

        pred_error = error(full_pd, pd_fn)
        errors.append(pred_error)
        if i % 20 == 0:
            plot(Xs, labels, new_point, None, full_pd)
            print(pred_error)

    plt.imshow(full_pd, origin="lower")
    plt.show()

    print(errors)
    print(len(errors))


if __name__ == "__main__":
    main()
