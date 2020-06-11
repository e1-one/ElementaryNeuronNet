# Elementary Neuron Net
> 👶 It is child's play with a self-written easy neural net 
>
> Created for educational purposes, fully autonomous (no dependencies; except numpy),
> a simple neural network that is used for image classification
>
This neural network has the next features:
 - feed-forward Neural Network or Artificial Neural Network (ANN) 
 - has a hidden layer
 - activation function: sigmoid
 - is fed with "full batch" (during one epoch weights adjustment is based on all training examples)
 - weights could be written/loaded to the file

####How to run it
*./download_training_data.sh* - downloads training set into /data folder
*./src/example_digits.py*  - demo of how this NN recognizes handwritten digits

#####Other:
*./src/helper.mnist* contains functions for operating with [MNIST](http://yann.lecun.com/exdb/mnist/) database