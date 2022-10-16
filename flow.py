# layers data are vertical -> so weight matrix are horizontal
#
#                z      
#   wwwwww       z       bias       z
#   wwwwww   *   z   +   bias   =   z
#   wwwwww       z       bias       z
#                z
#                z


import numpy as np

def rand_weights_bias(Ys_width, Xs_width):
    norm = Xs_width * Xs_width
    matrix_weights = np.random.randn(Ys_width, Xs_width) / norm
    bias = np.random.randn(Ys_width) / norm
    return matrix_weights, bias


def forward(Xs, matrix_weights, bias):
    Ys = np.dot(matrix_weights, Xs) + bias
    return Ys


def backward(delta_Ys, Xs, Xs_width, Xs_weights, Ys_width):
    # delta_Xs_weights = np.dot(delta_Ys, (Xs) / np.sum(Xs)) / Xs_width
    delta_Xs_weights = np.dot(delta_Ys.reshape(-1,1), (1 / Xs).reshape((1,-1))) / Xs_width
    
    delta_bias = sum(delta_Ys) / Xs_width
    # delta_Xs = np.dot(Xs_weights/np.sum(Xs_weights)) / Ys_width * (Xs > 0)
    delta_Xs = np.dot(Xs_weights.T, delta_Ys) / np.sum(Xs_weights.T, axis = 1) * (Xs > 0) / Ys_width

    return delta_Xs_weights, delta_bias, delta_Xs


def update_params(lr, Xs_weights, delta_Xs_weights, bias, delta_bias):
    Xs_weights -= delta_Xs_weights * lr
    bias -= delta_bias * lr
    return Xs_weights, bias