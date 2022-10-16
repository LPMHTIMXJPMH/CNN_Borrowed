import cvConfig
length = cvConfig.width
layers_length = len(length)
conns = layers_length - 1
weights = [None] * conns
bias = [None] * conns

layer_status = [None] * conns
# deltas = [None] * conns

import numpy as np

from rand_image import image
# None convolution layer has been applied :
image = image.flatten().T
image = image / 255

label_y_index = np.random.randint(10, dtype = np.uint8)
label_y = np.zeros(10, dtype = np.uint8)
label_y[label_y_index] = 1

import flow
import methods

# Initialization <=> weights and bias
for index in range(conns):
    weights[index], bias[index] = flow.rand_weights_bias(cvConfig.width[index + 1], cvConfig.width[index])

for i in range(2):
    # Forward flowing
    layer_status[0] = flow.forward(image, weights[0], bias[0])

    for index in range(conns - 1 - 1):
        layer_status[index + 1] = flow.forward(layer_status[index], weights[index + 1], bias[index + 1])
        layer_status[index + 1] = methods.relu(layer_status[index + 1])
    layer_status[-1] = np.dot(weights[-1], layer_status[-2]) + bias[-1]
    layer_status[-1] = methods.softmax(layer_status[-1])
    print(layer_status)
    loss = layer_status[-1] - label_y.T # methods.one_hot_y(label_y).T
    print(loss)

    # Backward flowing
    # It should be softmax <- not derivative of relu()
    delta_weights, delta_bias, delta = flow.backward(loss, layer_status[-2], len(layer_status[-2]), weights[-1], len(layer_status[-1]))
    weights[-1], bias[-1] = flow.update_params(cvConfig.learning_rate, weights[-1], delta_weights, bias[-1], delta_bias)

    for index in range(conns - 1):
        index = conns -1 - index - 1
        if index > 0:
            delta_weights, delta_bias, delta = flow.backward(delta, layer_status[index-1], len(layer_status[index-1]), weights[index], len(layer_status[index]))
            weights[index], bias[index] = flow.update_params(cvConfig.learning_rate, weights[index], delta_weights, bias[index], delta_bias)
        else:
            delta_weights, delta_bias, delta = flow.backward(delta, image, len(image), weights[index], len(layer_status[index]))
            weights[index], bias[index] = flow.update_params(cvConfig.learning_rate, weights[index], delta_weights, bias[index], delta_bias)

        print('weights')
        print(weights)
        print('bias')
        print(bias)