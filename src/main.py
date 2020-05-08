from neuron_net import ElementaryNeuronNet


nn = ElementaryNeuronNet(hidden_layer_neuron_count=3, output_layer_neurons_count=1, input_data_shape=3, epochs=1000)

X_train = [[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]]
y_train = [0, 1, 1, 0]

nn.train(X_train, y_train)

print(f"nn output {nn.think([0, 1, 0])}")
print(f"nn output {nn.think([1, 1, 1])}")
