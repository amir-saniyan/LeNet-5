In the name of God

# LeNet-5
This repository contains implementation of LeNet-5 (Handwritten Character Recognition) by Tensorflow and the network
tested with the [mnist dataset](http://yann.lecun.com/exdb/mnist/) and
[hoda dataset](http://farsiocr.ir/مجموعه-داده/مجموعه-ارقام-دستنویس-هدی).

![LeNet-5 Architecture](lenet.png)

# Training mnist dataset
To train the network with mnist dataset, type the following command at the command prompt:
```
python3 ./train_mnist.py
```

Sample images from mnist dataset:

![mnist sample](mnist_sample.png)

## Results

### Epoch 0
```
Train Accuracy = 0.081
Test Accuracy = 0.086
Validation Accuracy = 0.081
```

### Epoch 1
```
Train Accuracy = 0.963
Test Accuracy = 0.968
Validation Accuracy = 0.965
```

### Epoch 2
```
Train Accuracy = 0.978
Test Accuracy = 0.979
Validation Accuracy = 0.978
```

...

### Epoch 50
```
Train Accuracy = 0.999
Test Accuracy = 0.990
Validation Accuracy = 0.990
```

...

### Epoch 100
```
Final Train Accuracy = 1.000
Final Test Accuracy = 0.993
Final Validation Accuracy = 0.993
```

# Training hoda dataset
To train the network with hoda dataset, type the following command at the command prompt:
```
python3 ./train_hoda.py
```

Sample images from hoda dataset:

![hoda sample](hoda_sample.png)

## Results

### Epoch 0
```
Train Accuracy = 0.084
Test Accuracy = 0.086
Remaining Accuracy = 0.088
```

### Epoch 1
```
Train Accuracy = 0.984
Test Accuracy = 0.969
Remaining Accuracy = 0.973
```

### Epoch 2
```
Train Accuracy = 0.988
Test Accuracy = 0.977
Remaining Accuracy = 0.980
```

...

### Epoch 50
```
Train Accuracy = 1.000
Test Accuracy = 0.993
Remaining Accuracy = 0.993
```

...

### Epoch 100
```
Final Train Accuracy = 1.000
Final Test Accuracy = 0.993
Final Remaining Accuracy = 0.994
```

# Dependencies
* Python 3
* numpy
* python-opencv
* tensorflow

# Links
* http://yann.lecun.com/exdb/lenet/
* http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
* http://yann.lecun.com/exdb/mnist/
* http://farsiocr.ir/مجموعه-داده/مجموعه-ارقام-دستنویس-هدی
* http://dadegan.ir/catalog/hoda
* https://www.tensorflow.org/
* https://github.com/sujaybabruwad/LeNet-in-Tensorflow
* https://github.com/amir-saniyan/HodaDatasetReader
* https://github.com/amir-saniyan/LeNet-5
