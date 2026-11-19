import math
import random


def sigmoid(x):
    if x > 30.0:
        return 1.0
    if x < -30.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


class Neuron:

    def __init__(self, n_inputs):
        self.weights = [random.uniform(-1.0, 1.0) for _ in range(n_inputs)]
        self.bias = random.uniform(-1.0, 1.0)
        self.last_input = []
        self.last_output = 0.0
        self.delta = 0.0

    def activate(self, inputs):
        self.last_input = inputs
        s = self.bias
        for i in range(len(self.weights)):
            s += self.weights[i] * inputs[i]
        self.last_output = sigmoid(s)
        return self.last_output


class Layer:

    def __init__(self, n_neurons, n_inputs):
        self.neurons = [Neuron(n_inputs) for _ in range(n_neurons)]

    def forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.activate(inputs))
        return outputs


class NeuralNetwork:

    def __init__(self, sizes):
        self.sizes = list(sizes)
        self.layers = []
        for i in range(1, len(self.sizes)):
            self.layers.append(Layer(self.sizes[i], self.sizes[i - 1]))

    def count_weights(self):
        n = 0
        for layer in self.layers:
            for neuron in layer.neurons:
                n += len(neuron.weights) + 1
        return n

    def forward(self, x):
        data = x
        for layer in self.layers:
            data = layer.forward(data)
        return data

    def predict(self, x):
        outputs = self.forward(x)
        return outputs[0]

    def train(self, dataset, epochs, lr, log_every=20):
        history = []
        for epoch in range(1, epochs + 1):
            random.shuffle(dataset)
            total_error = 0.0
            for inputs, target in dataset:
                outputs = self.forward(inputs)
                for j in range(len(outputs)):
                    diff = target[j] - outputs[j]
                    total_error += diff * diff
                self.backward(target, lr)
            mse = total_error / len(dataset)
            history.append(mse)
            if log_every and (epoch == 1 or epoch % log_every == 0):
                print("  эпоха {:>4}/{} | средняя ошибка: {:.4f}".format(epoch, epochs, mse))
        return history

    def backward(self, target, lr):
        out_layer = self.layers[-1]
        for neuron in out_layer.neurons:
            y = neuron.last_output
            neuron.delta = (target[0] - y) * y * (1.0 - y)

        for li in range(len(self.layers) - 2, -1, -1):
            layer = self.layers[li]
            next_layer = self.layers[li + 1]
            for i, neuron in enumerate(layer.neurons):
                s = 0.0
                for next_neuron in next_layer.neurons:
                    s += next_neuron.delta * next_neuron.weights[i]
                y = neuron.last_output
                neuron.delta = s * y * (1.0 - y)

        for layer in self.layers:
            for neuron in layer.neurons:
                for i in range(len(neuron.weights)):
                    neuron.weights[i] += lr * neuron.delta * neuron.last_input[i]
                neuron.bias += lr * neuron.delta

    def export(self):
        layers_data = []
        for layer in self.layers:
            layer_data = []
            for neuron in layer.neurons:
                layer_data.append({"weights": neuron.weights, "bias": neuron.bias})
            layers_data.append(layer_data)
        return {"sizes": self.sizes, "layers": layers_data}

    @classmethod
    def from_dict(cls, data):
        net = cls(data["sizes"])
        for li, layer_data in enumerate(data["layers"]):
            for ni, neuron_data in enumerate(layer_data):
                net.layers[li].neurons[ni].weights = list(neuron_data["weights"])
                net.layers[li].neurons[ni].bias = neuron_data["bias"]
        return net
