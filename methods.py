import numpy as np

def relu(Ys):
    return np.maximum(0, Ys)

def softmax(Ys):
    # e_to_the_powers_of_y = []
    # for y in Ys:
    #     e_to_the_powers_of_y.append(np.exp(y))
    # sum = sum(e_to_the_powers_of_y)
    # return e_to_the_powers_of_y / sum
    return np.exp(Ys) / np.sum(np.exp(Ys))

def one_hot_y(Ys):
    # Ys.max() > Ys.size() beacause Ys.max() could be any number.
    one_hot = np.zeros(Ys.size(), Ys.max())
    one_hot[np.arange(Ys.size()), Ys] = 1
    return one_hot

def rand_weights():
    pass