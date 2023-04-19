# import pickle
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# def unpickle(file):
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#     return dict
# file_path = "../data/test_batch"
#
# data = unpickle(file_path)
#
# images = data[b'data']
# image = images[1]
# image = np.reshape(image, (32, 32, 3))
#
# plt.imshow(image)
# plt.show()

import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.datasets import cifar10

# load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# reshape the dataset
x_train_rgb = x_train.reshape((x_train.shape[0], 32, 32, 3))
x_test_rgb = x_test.reshape((x_test.shape[0], 32, 32, 3))

# print the new shape of the dataset
print("New shape of x_train_rgb:", x_train_rgb.shape)
print("New shape of x_test_rgb:", x_test_rgb.shape)

test = x_train_rgb[3]
plt.imshow(test)
plt.show()
