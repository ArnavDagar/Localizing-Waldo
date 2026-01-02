import numpy as np
import tensorflow as tf
import pandas as pd
import cv2 as cv
#import yaml
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer

def normalize_image(image_data):
    """
    Normalize the image data with Min-Max scaling to a range of [0.1, 0.9]
    :param image_data: The image data to be normalized - it expects values between 0,255
    :return: Normalized image data
    """
    a = 0.1
    b = 0.9
    grayscale_min = 0
    grayscale_max = 255
    return a + ( ( (image_data - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )

# Read the images and labels
# Seperate out the test data = Every 5th File
# Rest is training
# Images are already undistorted, but in color

fileStr = 'IDLabels.csv'
df = pd.read_csv(fileStr)

all_idx = np.arange(0,5196)
test_idx = all_idx%5 == 0
test_data_imgs = df['ImgID'][test_idx]
y_test = df['Label'][test_idx]
X_test = []
train_idx = all_idx%5 != 0
train_data_imgs = df['ImgID'][train_idx]
y_train = df['Label'][train_idx]
X_train = []

print('Load Training Images')

for idx,imgid in enumerate(train_data_imgs):
    if idx % 100 == 0:
        print(idx)
    #print(idx)
    img = cv.imread('LandmarkIDImages/' + imgid + '.jpg')
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    X_train.append(img)
X_train = np.array(X_train)

print('Read Training Images')

for imgid in test_data_imgs:
    img = cv.imread('LandmarkIDImages/' + imgid + '.jpg')
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    X_test.append(img)
X_test = np.array(X_test)

print('Read Test Images')

# Setup Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

# Build the Final Test Neural Network in Keras Here
model = Sequential()
model.add(Conv2D(8,kernel_size=(7,7),padding='VALID',input_shape=(64,64,1),
                 activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(24,kernel_size=(5,5),padding='VALID',input_shape=(32,32,1),
                 activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(144,kernel_size=(3,3),padding='VALID',input_shape=(16,16,1),
                 activation='relu'))
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5,activation='softmax'))

# preprocess data
# Normalize all the images
X_train = normalize_image(X_train)
X_test = normalize_image(X_test)

# Shuffle the training data
X_train, y_train = shuffle(X_train, y_train)


label_binarizer = LabelBinarizer()
y_one_hot = label_binarizer.fit_transform(y_train)
'''

# compile and fit the model
model.compile('adam', 'categorical_crossentropy', ['accuracy'])
history = model.fit(X_train, y_one_hot, epochs=10, validation_split=0.2)

print("Testing")

# preprocess data
y_one_hot_test = label_binarizer.fit_transform(y_test)

metrics = model.evaluate(X_test, y_one_hot_test)
for metric_i in range(len(model.metrics_names)):
    metric_name = model.metrics_names[metric_i]
    metric_value = metrics[metric_i]
    print('{}: {}'.format(metric_name, metric_value))

model.save('test_model.h5')
'''
