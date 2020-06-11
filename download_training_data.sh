#!/bin/bash

echo "Downloading MNIST handwritten digits databese"
``
DATA_PATH='data/digits'
echo "Data path is $DATA_PATH"
mkdir $DATA_PATH

curl -o "$DATA_PATH/train-images-idx3-ubyte.gz" http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
curl -o "$DATA_PATH/train-labels-idx1-ubyte.gz" http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
curl -o "$DATA_PATH/t10k-images-idx3-ubyte.gz"  http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
curl -o "$DATA_PATH/t10k-labels-idx1-ubyte.gz"  http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

echo "Files were created:"
find $DATA_PATH -type f
