"""
    Run t-SNE on the nxd array X to reduce its dimensionality to no_dims

    Usage:
        tsne.py <input_file> <output_file> [--dim=<dim>] [--perplexity=<perplexity>] [--max_iter=<max_iter>] [--plot] [--initial_momentum=<initial_momentum>] [--final_momentum=<final_momentum>] [--eta=<eta>] [--min_gain=<min_gain>] [--tol=<tol>]

    Options:
        -h --help  Show this screen.
        --dim=<dim>  number of dimensions that t-SNE reduce to [default: 2]
        --perplexity=<perplexity>  perplexity can be interpreted as a smooth measure of the effective number of neighbors [default: 30.0]
        --max_iter=<max_iter>  maximum number of iterations [default: 1000]
        --plot  whether to plot the result
        --initial_momentum=<initial_momentum>  initial momentum to be used in the gradient descent for first 20 iterations [default: 0.5]
        --final_momentum=<final_momentum>  final momentum to be used in the gradient descent after 20 iterations [default: 0.8]
        --eta=<eta>  learning rate for gradient descent [default: 500]
        --min_gain=<min_gain>  minimum gain for gradient descent [default: 0.01]
        --tol=<tol>  tolerance for the stopping criteria of beta adjustment [default: 1e-5]
"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from docopt import docopt


def pca(X, no_dims=50):
    """
    Runs PCA on the nxd array X in order to reduce its dimensionality to
    no_dims dimensions.

    Parameters
    ----------
    X : numpy.ndarray
        data input array with dimension (n,d)
    no_dims : int
        number of dimensions that PCA reduce to

    Returns
    -------
    Y : numpy.ndarray
        low-dimensional representation of input X
    """
    n, d = X.shape
    X = X - X.mean(axis=0)[None, :]
    _, M = np.linalg.eig(np.dot(X.T, X))
    Y = np.real(np.dot(X, M[:, :no_dims]))
    return Y


def Hbeta(D, beta=1.0):
    """
    Compute entropy(H) and probability(P) from nxn distance matrix.

    Parameters
    ----------
    D : numpy.ndarray
        distance matrix (n,n)
    beta : float
        precision measure
    .. math:: \beta = \frac{1}/{(2 * \sigma^2)}

    Returns
    -------
    H : float
        entropy
    P : numpy.ndarray
        probability matrix (n,n)
    """
    num = np.exp(-D * beta)
    den = np.sum(np.exp(-D * beta), 0)
    P = num / den
    H = np.log(den) + beta * np.sum(D * num) / (den)
    return H, P


def adjustbeta(X, D, tol, perplexity):
    """
    Precision(beta) adjustment based on perplexity

    Parameters
    ----------
    X : numpy.ndarray
        data input array with dimension (n,d)
    D : numpy.ndarray
        distance matrix (n,n)
    tol : float
        tolerance for the stopping criteria of beta adjustment
    perplexity : float
        perplexity can be interpreted as a smooth measure of the effective number of neighbors

    Returns
    -------
    P : numpy.ndarray
        probability matrix (n,n)
    beta : numpy.ndarray
        precision array (n,1)
    """
    (n, d) = X.shape
    # Need to compute D here, which is nxn distance matrix of X
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i + 1 : n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:
            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.0
                else:
                    beta[i] = (beta[i] + betamax) / 2.0
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.0
                else:
                    beta[i] = (beta[i] + betamin) / 2.0

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i + 1 : n]))] = thisP

    return P, beta


def tsne(
    X,
    dim=2,
    perplexity=30.0,
    max_iter=1000,
    initial_momentum=0.5,
    final_momentum=0.8,
    eta=500,
    min_gain=0.01,
    tol=1e-5,
):
    """
    Run t-SNE on the nxd array X to reduce its dimensionality to no_dims
    Parameters
    ----------
    X : numpy.ndarray
        data input array with dimension (n,d)
    dim : int, optional
        number of dimensions that t-SNE reduce to
    perplexity : float, optional
        perplexity can be interpreted as a smooth measure of the effective number of neighbors
    max_iter : int, optional
        maximum number of iterations
    initial_momentum : float, optional
        initial momentum to be used in the gradient descent for first 20 iterations
    final_momentum : float, optional
        final momentum to be used in the gradient descent after 20 iterations
    eta : float, optional
        learning rate for gradient descent
    min_gain : float, optional
        minimum gain for gradient descent
    tol : float, optional
        tolerance for the stopping criteria of beta adjustment

    Returns
    -------
    Y : numpy.ndarray
        low-dimensional representation of input X
    """
    D = (X**2).sum(axis=1)[None, :] - 2 * X @ (X.T) + (X**2).sum(axis=1)[:, None]
    Pji, beta = adjustbeta(X, D, tol, perplexity)
    Pij = Pji + Pji.T
    Pij = Pij / np.sum(Pij)
    # exagerate Pij for early iteration
    Pij = np.maximum(Pij, 1e-12) * 4.0

    Y = X[:, :dim]
    n = Y.shape[0]
    dY = np.zeros((n, dim))
    iY = np.zeros((n, dim))
    gains = np.ones((n, dim))
    for t in tqdm(range(max_iter)):
        dist_Y = (
            (Y**2).sum(axis=1)[None, :] - 2 * Y @ (Y.T) + (Y**2).sum(axis=1)[:, None]
        )
        num = 1 / (1 + dist_Y)
        num[np.arange(num.shape[0]), np.arange(num.shape[0])] = 0
        Qij = num / np.sum(num)
        Qij = np.maximum(Qij, 1e-12)

        kl = Pij * np.log(Pij / Qij)
        kl[np.isnan(kl)] = 0
        kl = np.sum(kl)
        if (t % 100) == 0:
            print(f"Iteration {t}: KL = {kl}")

        PQ = Pij - Qij

        for i in range(n):
            dY[i, :] = np.sum(
                np.tile(PQ[:, i] * num[:, i], (dim, 1)).T * (Y[i, :] - Y), 0
            )

        if t < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum

        gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * (
            (dY > 0) == (iY > 0)
        )
        gains[gains < min_gain] = min_gain

        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))
        if t == 100:
            Pij = Pij / 4.0

    return Y


if __name__ == "__main__":
    args = docopt(__doc__)
    X = np.loadtxt(args["<input_file>"])
    X = pca(X, 50)
    print(args["--plot"])
    Y = tsne(
        X,
        int(args["--dim"]),
        float(args["--perplexity"]),
        int(args["--max_iter"]),
        float(args["--initial_momentum"]),
        float(args["--final_momentum"]),
        float(args["--eta"]),
        float(args["--min_gain"]),
        float(args["--tol"]),
    )
    if bool(args["--plot"]):
        plt.scatter(Y[:, 0], Y[:, 1])
        plt.savefig("tsne.png")
    np.save(args["<output_file>"], Y)