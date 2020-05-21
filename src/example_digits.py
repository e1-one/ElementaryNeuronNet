import matplotlib.pyplot as plt
import numpy as np

from helper.mnist import mnist_helper, mnist_reader
from helper.nn_persist import write_weights_as_json_to_file, init_nn_with_weights_from_file
from neuron_net_2 import ElementaryNeuronNet2

X_train, y_train = mnist_reader.load_mnist('../data/digits', kind='train')
X_test, y_test = mnist_reader.load_mnist('../data/digits', kind='t10k')
labels = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

# input=500 1hl=40 2hl=4 lr=0.00005        epoch: 10200 from 100000  Error: 0.07709423218073314
# input=500 1hl=40 2hl=4 lr=0.000005
# input=1000 1hl=10 2hl=3 lr=0.0001        epoch: 10200 from 100000  Error: 0.07709423218073314
# input=500 1hl=50 2hl=5 lr=0.00005        epoch: 3800 from 100000  Error: 0.08403416115152
# input=500 1hl=80 2hl=10 lr=0.00005       epoch: 3300 from 100000  Error: 0.08863587289836697
# input=500 1hl=80 2hl=10 lr=0.00001       epoch: 3300 from 100000  Error: 0.08863587289836697

offset = 2000
input_data_size = 300
epochs = 10000
nn = ElementaryNeuronNet2(first_hidden_layer_neuron_count=20,
                          second_hidden_layer_neuron_count=5,
                          output_layer_neurons_count=1,
                          input_data_shape=28 * 28,
                          min_step=0.0001)
nn_weights_file_name = 'nn_weights.json'
X_train_nn_input = np.split(X_train, [offset, offset + input_data_size, ])[1] / 255
y_train_nn_input = np.split(y_train, [offset, offset + input_data_size, ])[1] / 10

print('Do you want to train the neural net or just load the weights?')
ans = input('(T (train)/ L (load)) << ').lower()
if ans in ['train', 't']:
    nn.train(X_train_nn_input, y_train_nn_input, epochs=epochs)
    write_weights_as_json_to_file(nn, nn_weights_file_name)
else:
    nn = init_nn_with_weights_from_file(nn_weights_file_name)


def convert_nn_out_to_human_readable_text(net_output):
    raw_net_output = net_output[0]
    print(f"raw_net_output {raw_net_output}")
    index = int(round(raw_net_output, 1) * 10)
    return labels[index]


def show_pictures_with_titles(pictures, data_labels=None):
    for i in range(len(pictures)):
        data = pictures[i]
        answer = convert_nn_out_to_human_readable_text(nn.think(data))
        if data_labels is not None and data_labels.any():
            print(f"actual data is {data_labels[i]}")
        plt.gcf().canvas.set_window_title(f"neuron net says that is {answer} ")
        plt.imshow(mnist_helper.get_sprite_image(data), cmap='gray', interpolation="bicubic")
        plt.show()


show_pictures_with_titles(X_train_nn_input, y_train_nn_input)
