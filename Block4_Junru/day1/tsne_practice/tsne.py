#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：convnet_prize 
@File    ：tsne.py
@Author  ：Junru Jin
@Date    ：5/28/24 2:29 PM 
'''
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

sys.path.append('/archive/bioinformatics/Zhou_lab/shared/jjin/project/convnet_prize/high_dimension/day1')

from adjustbeta import Hbeta, adjustbeta


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


def tsne(X, no_dims, tol, perplexity, step_size, min_gain, initial_momentum, final_momentum, eta, T):
    """
    Computes low-dimentional representation of inuput data matrix X

    Parameters
    ----------
    X: numpy.ndarray
       input data matrix after the linear dimentionality reduction
    no_dimen: int
       number of dimention that t-SNE reduces to
    tol : float
        tolerance for the stopping criteria of beta adjustment
    perplexity : float
        perplexity can be interpreted as a smooth measure of the effective number
        of neighbors
    step_size: int
        step size that the algorythm takes
    min_gain:float
        minimum gain that is used to clip the gain values
    initial_momentum: float
        momentum used for delta Y update at the first stages of training
    final_momentum: float
        momentum used for delta Y update at the later stages of training
    eta: float
        learning rate
    T: int
        number of iteration that are used toi train the model
    Returns
    -------
    Y: numpy.ndarray
       low dimentional representation of X, an array with dimentions (n, no_dimen)
    """
    (n, d) = X.shape

    # compute p_ij distribution and beta
    p_i_given_j, beta = adjustbeta(X, tol, perplexity)

    # compute p_ij distribution
    P = (p_i_given_j + p_i_given_j.T) / np.sum((p_i_given_j + p_i_given_j.T))
    P = P * 4
    P = np.clip(P, a_min=1e-12, a_max=None)

    # initialize variables
    Y = pca(X, no_dims)

    # Initialize ∆Y (n×no dims) = 0, gains(n×no dims) = 1
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # run t-SNE
    for t in tqdm(range(1, T)):
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        PQ = P - Q

        for i in range(n):
            dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

        if t < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum

        gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * ((dY > 0) == (iY > 0))
        gains[gains < min_gain] = min_gain

        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        if t == 100:
            P = P / 4.

    return Y


if __name__ == "__main__":
    print("Run Y = tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
    print("Running example on 2,500 MNIST digits...")
    X = np.loadtxt("mnist2500_X.txt")
    X = pca(X, 50)
    labels = np.loadtxt("mnist2500_labels.txt")
    Y = tsne(X, 2, 1e-5, 30.0, 500, 0.01, 0.5, 0.8, 500, 1000)
    plt.scatter(Y[:, 0], Y[:, 1], 20, labels)
    plt.savefig("mnist_tsne.png")
