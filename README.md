In the name of God

# LeNet-5
This repository contains implementation of LeNet-5 (Handwritten Character Recognition) by Tensorflow
and the network tested with the MNIST dataset.

![LeNet-5 Architecture](lenet.png)

# Testing Network
To test the network type the following command at the command prompt:
```
python3 ./handwritten_character_recognition.py
```

## Dependencies
* Python 3
* Tensorflow
* Numpy

# Links
* http://yann.lecun.com/exdb/lenet/
* http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
* http://yann.lecun.com/exdb/mnist/
* https://www.tensorflow.org/
* https://github.com/sujaybabruwad/LeNet-in-Tensorflow

# Results

## Epoch 0
```
Train Accuracy = 0.122
Validation Accuracy = 0.115
Test Accuracy = 0.116
```

## Epoch 1
```
Train Accuracy = 0.954
Validation Accuracy = 0.958
Test Accuracy = 0.958
```

## Epoch 2
```
Train Accuracy = 0.973
Validation Accuracy = 0.972
Test Accuracy = 0.973
```

...

## Epoch 50
```
Train Accuracy = 0.999
Validation Accuracy = 0.990
Test Accuracy = 0.991
```

...

## Epoch 100
```
Final Train Accuracy = 1.000
Final Validation Accuracy = 0.992
Final Test Accuracy = 0.993
```
