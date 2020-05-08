import numpy as np


class ElementaryNeuronNet:

    def __init__(self, hidden_layer_neuron_count, output_layer_neurons_count, input_data_shape, epochs=50000):
        np.random.seed(1)
        self.epochs = epochs
        self.input_data_shape = input_data_shape
        self.weights_on_hidden = ElementaryNeuronNet.random_weights(input_data_shape, hidden_layer_neuron_count)
        self.weights_final = ElementaryNeuronNet.random_weights(hidden_layer_neuron_count, output_layer_neurons_count)

    @staticmethod
    def random_weights(n, m):
        return np.array(2 * np.random.rand(n, m) - 1)

    @staticmethod
    def activation_function(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def derivative(x):
        return x * (1 - x)

    def train(self, train_data, expected_outcome: list):
        train_data = np.array(train_data)
        assert train_data.shape[1] == self.input_data_shape
        assert len(expected_outcome) == len(train_data)
        print("Start training")
        print(f"training data set: {len(train_data)}")
        expected_outcome = np.array([expected_outcome]).T
        for epoch in range(0, self.epochs + 1):
            comb_strength_on_hidden_neurons = self.activation_function(np.dot(train_data, self.weights_on_hidden))
            comb_strength_on_last_neuron = self.activation_function(np.dot(comb_strength_on_hidden_neurons, self.weights_final))
            error_last = expected_outcome - comb_strength_on_last_neuron
            slope = self.derivative(comb_strength_on_last_neuron)
            # confident errors are amplified, not confident are neglected
            delta_last = error_last * slope
            adjustment_weights_on_last = np.dot(comb_strength_on_hidden_neurons.T, delta_last)
            self.weights_final += adjustment_weights_on_last
            # "back propagation" algorithm,
            # contribution weighted error to the neurons the neurons that "contributed" to the next layer error
            error_hidden = np.dot(delta_last, self.weights_final.T)
            delta_on_hidden = error_hidden * self.derivative(comb_strength_on_hidden_neurons)
            adjustment_on_hidden_neurons = np.dot(train_data.T, delta_on_hidden)
            self.weights_on_hidden += adjustment_on_hidden_neurons

            if epoch % 1000 == 0 or epoch == self.epochs - 1:
                # print(f"epoch: {epoch} from {self.iterations_count}  delta_last: {np.mean(np.abs(delta_last))}")
                print(f"epoch: {epoch} from {self.epochs}  Error: {np.mean(np.abs(error_last))}")
        print("training finished")

    def think(self, data):
        assert len(data) == self.input_data_shape
        comb_strength_on_hidden_neurons = ElementaryNeuronNet.activation_function(np.dot(data, self.weights_on_hidden))
        comb_strength_on_last = self.activation_function(np.dot(comb_strength_on_hidden_neurons, self.weights_final))
        return comb_strength_on_last


