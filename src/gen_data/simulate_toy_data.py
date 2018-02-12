import numpy as np
import gpflow as gpflow
import matplotlib.pyplot as plt
from sklearn.externals import joblib


def gen_toy_data(xmin=0, xmax=+6, D=2, obs_dim=15, num_data=100):
    """
    :param D: latent space dimensionality
    :param xmin: function domain min
    :param xmax: function domain max
    :param obs_dim: dimensionality of observed data
    :param num_data: number of sequences
    :return: {x num_data * obs_dim * num_evals, f (latent functions)}
    """
    input_dim = 1
    kernels = {}
    num_evals = 45 # number of function evaluations
    x = np.zeros([num_data, obs_dim, num_evals])
    f = np.zeros([D, num_evals])
    p = np.zeros([3, num_evals])
    for d in np.arange(D):
        if d == 0:
            kernels[d] = gpflow.kernels.RBF(input_dim=input_dim, variance=1.0,
                                            lengthscales=9 * np.ones([input_dim]))
        elif d == 1:
            kernels[d] = gpflow.kernels.Cosine(input_dim=input_dim, variance=0.75,
                                               lengthscales=3 * np.ones([input_dim]))
            # kernels[d] = gpflow.kernels.Periodic(input_dim=input_dim, lengthscales=1.0*np.ones([input_dim]),
            #                                      period=0.5*np.ones([input_dim]))

    # define function domain
    xx = np.linspace(xmin, xmax, num_evals)[:, None]

    for i in np.arange(num_data):
        # sample GP functions
        for d in np.arange(D):
            K = kernels[d].compute_K_symm(xx)
            f[d] = np.random.multivariate_normal(np.zeros(xx.shape[0]), K, 1).ravel()
        # sample observations
        max_f = np.max(f, axis=0)
        shifted_f = f - max_f
        p[:2, :] = np.exp(shifted_f) / (np.sum(0.1 + np.exp(shifted_f), axis=0))
        p[2] = 1 - (p[0] + p[1])
        # map p to the observed space (p[0] governs the first obs_dim/2 observations, p[1] governs the remaining).
        data_p = np.zeros([obs_dim, num_evals])
        m = np.int(obs_dim / 3)
        data_p[:m, :] = p[0]
        data_p[m:2 * m, :] = p[1]
        data_p[2 * m:, :] = p[2]
        # Bernoulli distributed observations
        x[i] = np.random.binomial(1, data_p)
        num_hidden = np.random.poisson(lam=0.7*num_evals)
        mask = np.random.choice(num_evals, size=num_hidden)
        x[i, :, mask] = -1

    for d in np.arange(D):
        plt.plot(xx, f[d])
    plt.plot(xx, p[0], 'r--')
    plt.plot(xx, p[1], 'b--')
    plt.plot(xx, p[2], 'g--')
    # display a random sequence sequence
    plt.matshow(x[np.random.randint(num_data)])
    plt.show(block=True)
    return {'x': x, 'f': f, 'time': xx, 'p': p}


if __name__=='__main__':
    data = gen_toy_data(xmax=60, num_data=10000)
    savename = "../../data/toy_data_v3.pkl"
    joblib.dump(data, savename)
