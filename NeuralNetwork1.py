import numpy as np
import random

weights = [random.random(),
           random.random(),
           random.random(),
           random.random(),
           random.random(),
           random.random(),
           random.random(),
           random.random(),
           random.random(),
           random.random(),
           random.random(),
           random.random(),
           random.random(),
           random.random(),
           random.random(),
           random.random()]

biases = [random.random(),
        random.random(),
        random.random(),
        random.random(),
        random.random(),
        random.random(),
        random.random(),
        random.random()]

X = np.linspace(-5, 5, 1)
Y = 0.5 * X**3


def forward_prop():
    hidden_layer = []
    sums = 0

    for i in range(0, 8):
        hidden_layer.append(weights[i] * X[0] + biases[i])

    hidden_layer_sigmoid = sigmoid(hidden_layer)

    for i in range(0, 8):
        sums += weights[i + 8] * hidden_layer_sigmoid[i]

    return sums, hidden_layer_sigmoid


def sigmoid(hidden_layer):
    for i in range(0, 8):
        hidden_layer[i] = 1/(1+np.exp(-hidden_layer[i]))

    return hidden_layer

def loss_calc(out):
    print(1/2 * (out - Y)**2)

def back_prop(out, hidden_layers):
    dLdz = out - Y[0]
    dLdw2 = []
    dLdw1 = []
    dLdb1 = []
    for i in range(0, 8):
        dLdw2.append(dLdz * hidden_layers[i])
        dLdw1.append(dLdz * weights[i + 8] * hidden_layers[i] * (1 - hidden_layers[i]) * X[0])
        dLdb1.append(dLdz * weights[i + 8] * hidden_layers[i] * (1 - hidden_layers[i]))

    for i in range(0, 8):
        weights[i] -= 0.1 * dLdw1[i]
        weights[i + 8] -= 0.1 * dLdw2[i]
        biases[i] -= 0.1 * dLdb1[i]

output, hidden_layers = forward_prop()
loss_calc(output)

for i in range(100):
    back_prop(output, hidden_layers)
    output, hidden_layers = forward_prop()
loss_calc(output)
# print(output, hidden_layers)

print(output, Y[0])