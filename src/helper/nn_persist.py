import json

import numpy as np

from neuron_net_2 import ElementaryNeuronNet2


def write_as_json_to_file(nn, file_name):
    obj = {
        'class_version': nn.class_version,
        'weights_on_hidden.shape': nn.weights_on_hidden.shape,
        'weights_on_hidden_2.shape': nn.weights_on_hidden_2.shape,
        'weights_final.shape': nn.weights_final.shape,
        'weights_on_hidden': nn.weights_on_hidden.tolist(),
        'weights_on_hidden_2': nn.weights_on_hidden_2.tolist(),
        'weights_final': nn.weights_final.tolist()
    }
    with open(file_name, 'w') as outfile:
        json.dump(obj, outfile)


def load_from_file(file_name):
    with open(file_name, 'r') as f:
        import json
        obj = json.load(f)
    print(f"loaded {obj}")
    assert obj[
               'class_version'] == ElementaryNeuronNet2.class_version, "version of the loaded object differs from the existing one "

    nn = ElementaryNeuronNet2(
        input_data_shape=obj['weights_on_hidden.shape'][0],
        first_hidden_layer_neuron_count=obj['weights_on_hidden.shape'][1],
        second_hidden_layer_neuron_count=obj['weights_on_hidden_2.shape'][0],
        output_layer_neurons_count=obj['weights_final.shape'][1]
    )
    nn.weights_on_hidden = np.array(obj['weights_on_hidden'])
    nn.weights_on_hidden_2 = np.array(obj['weights_on_hidden_2'])
    nn.weights_final = np.array(obj['weights_final'])
    print("NN Weights are loaded from the file.")
    return nn
