import numpy as np


class ElementaryNeuronNet2:
    class_version = 1

    def __init__(self, first_hidden_layer_neuron_count,
                 second_hidden_layer_neuron_count, output_layer_neurons_count, input_data_shape, min_step=0.0001):
        np.random.seed(1)
        self.input_data_shape = input_data_shape
        self.weights_on_hidden = self.random_weights(input_data_shape, first_hidden_layer_neuron_count)
        self.weights_on_hidden_2 = self.random_weights(first_hidden_layer_neuron_count, second_hidden_layer_neuron_count)
        self.weights_final = ElementaryNeuronNet2.random_weights(second_hidden_layer_neuron_count,
                                                                 output_layer_neurons_count)
        self.min_step = min_step

    @staticmethod
    def random_weights(n, m):
        # return np.zeros((n,m))
        return np.array(2 * np.random.rand(n, m) - 1)

    @staticmethod
    def activation_function(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def derivative(x):
        return x * (1 - x)

    def limit_function(self, x: np.ndarray):
        vectorized_limit = np.vectorize(self.limit_minimum_change_rate)
        return vectorized_limit(x)

    def limit_minimum_change_rate(self, change_rate):
        if change_rate < -self.min_step:
            return -self.min_step
        elif change_rate < self.min_step:
            return self.min_step
        else:
            return change_rate

    def train(self, train_data, expected_outcome: list, epochs=10000):
        train_data = np.array(train_data)
        expected_outcome = np.array([expected_outcome]).T
        self.validate_input(expected_outcome, train_data)
        print("Start training")
        print(f"training data set: {len(train_data)}")
        error_prev = 1  # initialize error with max value
        for epoch in range(0, epochs + 1):
            comb_strength_on_hidden_neurons = self.activation_function(np.dot(train_data, self.weights_on_hidden))
            comb_strength_on_hidden_neurons_2 \
                = self.activation_function(np.dot(comb_strength_on_hidden_neurons, self.weights_on_hidden_2))
            comb_strength_on_last_neuron = self.activation_function(
                np.dot(comb_strength_on_hidden_neurons_2, self.weights_final))

            error_last = expected_outcome - comb_strength_on_last_neuron
            slope = self.derivative(comb_strength_on_last_neuron)
            # confident errors are amplified, not confident are neglected
            delta_last = error_last * slope
            adjustment_weights_on_last = self.limit_function(np.dot(comb_strength_on_hidden_neurons_2.T, delta_last))
            self.weights_final += adjustment_weights_on_last
            # "back propagation" algorithm,
            # contribution weighted error to the neurons the neurons that "contributed" to the next layer error
            error_hidden_2 = np.dot(delta_last, self.weights_final.T)
            delta_on_hidden_2 = error_hidden_2 * self.derivative(comb_strength_on_hidden_neurons_2)
            adjustment_on_hidden_neurons_2 = self.limit_function(
                np.dot(comb_strength_on_hidden_neurons.T, delta_on_hidden_2))
            self.weights_on_hidden_2 += adjustment_on_hidden_neurons_2

            error_hidden = np.dot(delta_on_hidden_2, self.weights_on_hidden_2.T)
            delta_on_hidden = error_hidden * self.derivative(comb_strength_on_hidden_neurons)
            adjustment_on_hidden_neurons = self.limit_function(np.dot(train_data.T, delta_on_hidden))
            self.weights_on_hidden += adjustment_on_hidden_neurons
            if (0 < epoch < 0) or epoch % 100 == 0 or epoch == epochs - 1:
                # print(f"epoch: {epoch} from {self.iterations_count}  delta_last: {np.mean(np.abs(delta_last))}")
                error = np.mean(np.abs(error_last))
                error_decreasing_temp = error - error_prev
                print(f"epoch: {epoch} from {epochs}  Error: {error} Error decreasing: {error_decreasing_temp}")
                if error_decreasing_temp > 0:
                    # print("error is increasing now. No sense to continue evaluations. Stop processing")
                    break
                    pass
                error_prev = error
        print("training finished")
        print(f"epoch: {epoch} from {epochs}  Error: {np.mean(np.abs(error_last))}")

    def validate_input(self, expected_outcome, train_data):
        assert train_data.shape[1] == self.input_data_shape, "input data shape differs from the one used for training"
        assert expected_outcome.shape[0] == train_data.shape[0], "labels are not matching to the data"

    def think(self, data):
        assert len(data) == self.input_data_shape, "input data's shape differs from the one used for model training"
        comb_strength_on_hidden_neurons_1 = self.activation_function(np.dot(data, self.weights_on_hidden))
        comb_strength_on_hidden_neurons_2 = self.activation_function(
            np.dot(comb_strength_on_hidden_neurons_1, self.weights_on_hidden_2))
        comb_strength_on_last = self.activation_function(np.dot(comb_strength_on_hidden_neurons_2, self.weights_final))
        return comb_strength_on_last

