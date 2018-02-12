import numpy as np
from sklearn.externals import joblib


def kernel_function(t1, t2):
    noise = 1e-3
    if t1 != t2:
        noise = 0.0
    signal = 1.0 - noise
    time_char = 10.0
    return signal * np.exp(-(t1-t2)**2)/(2*time_char**2)
    #return signal * np.exp(-np.power((float(t1) - float(t2)), 2) / (2.0 * np.power(time_char, 2))) + noise


def kernel_matrix(sequence1, sequence2):
    k_mat = np.zeros((len(sequence1), len(sequence2)), dtype=np.float32)
    ax_1, ax_2 = k_mat.shape
    for i, ele1 in enumerate(sequence1):
        for j, ele2 in enumerate(sequence2):
            print(ele1, ele2)
            k_mat[i, j] = kernel_function(ele1, ele2)
    return k_mat


def main():
    data = joblib.load('../data/toy_data.pkl')
    data = np.reshape(data['time'][np.where(data['x'][0, 0, :] > -1)], [1, -1])
    test = kernel_matrix(data[0]*1e+2, data[0]*1e+2)
    # print(data[0])
    # print(test)
    print(np.linalg.det(test))
    a = np.linalg.cholesky(test)
    print (a)


if __name__ == '__main__':
    main()
