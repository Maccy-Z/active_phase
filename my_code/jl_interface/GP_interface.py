from GP_interface2 import JuliaGP

import numpy as np
from matplotlib import pyplot as plt


def generate_cluster(center, n_points=100, std_dev=0.5):
    """Generate a cluster of points."""
    return np.random.normal(center, std_dev, (n_points, 2))


def gen_data():
    n_points_per_cluster = 100
    coords = [[1, 1], [0, 0], [-1, -1]]
    # Generate 2D dataset
    clusters, labels = [], []
    for label, coord in enumerate(coords):
        clusters.append(generate_cluster(coord, n_points=n_points_per_cluster))
        labels.append(np.ones(n_points_per_cluster) * label + 1)

    # Combine the datasets
    X = np.vstack(clusters)
    y = np.concatenate(labels)

    return X, y


def test_data(n_points):
    x_values = np.linspace(-3, 3, n_points)  # 100 points from -3 to 3 for the x-axis
    y_values = np.linspace(-3, 3, n_points)  # 100 points from -3 to 3 for the y-axis

    # Create a 2D grid of points
    xx, yy = np.meshgrid(x_values, y_values)
    x_flat = xx.ravel()
    y_flat = yy.ravel()
    points = np.vstack((x_flat, y_flat)).T

    # Convert to a list of tuples (optional)
    # points_list = [tuple(point) for point in points]
    return points


def main():
    n_points = 50
    X, y = gen_data()
    X_test = test_data(n_points)
    n_samples = 400
    full_cov = False
    julia_gp = JuliaGP()

    julia_gp.make_sampler(len(X_test) * n_samples)
    print("sampler made")

    # Make the Gaussian process
    julia_gp.make_GP(X, y, n_class=3, init_sigma=1., init_scale=1., optimiser="ADAM")
    print("GP made")

    # Train the GP
    vars, inv_lens = julia_gp.train_GP(n_iter=100)
    print("Trained")

    # Print kernel params
    for var, inv_len in zip(vars, inv_lens):
        print(f"variance: {var:.3g}, lengthscale: {1 / inv_len :.3g}")

    # Predict the labels for the test data
    (probs, p_std), (f_mu, f_var) = julia_gp.pred_proba_sampler(X_test, full_cov=full_cov, model_cov=True, nSamples=n_samples)

    """Plotting"""
    f_mu, f_std = np.array(f_mu), np.sqrt(np.abs(np.array(f_var)))
    probs, p_std = np.array(probs), np.array(p_std)
    f_mu, f_std = f_mu[1], f_std[1]
    probs, p_std = probs[:, 1], p_std[:, 1]

    if full_cov:
        f_std = np.diag(f_std)
        print(f_std.shape)

    # Show the probs
    plt.subplot(1, 2, 1)
    plt.imshow(f_mu.reshape((n_points, n_points)), origin="lower", extent=(-3, 3, -3, 3))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=4)

    plt.subplot(1, 2, 2)
    plt.imshow(probs.reshape((n_points, n_points)), origin="lower", extent=(-3, 3, -3, 3))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=4)
    plt.show()

    # Show the stds
    plt.subplot(1, 2, 1)
    plt.imshow(f_std.reshape((n_points, n_points)), origin="lower", extent=(-3, 3, -3, 3))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=4)

    plt.subplot(1, 2, 2)
    plt.imshow(p_std.reshape((n_points, n_points)), origin="lower", extent=(-3, 3, -3, 3))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=4)
    plt.show()


if __name__ == "__main__":
    main()
