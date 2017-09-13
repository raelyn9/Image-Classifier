import numpy as np
import glob
import scipy.misc
import cv2

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
import matplotlib.pyplot as plt
from keras import backend as K
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

train_data_dir = './train_data/*.jpeg'
validation_data_dir = './validation_data/*.jpeg'

WIDTH = 100
HEIGHT = 55
input_shape = (HEIGHT, WIDTH, 3)

#add object type
object_types = []

def read_data(dirpath):

    filenames = glob.glob(dirpath)
    images = np.empty([len(filenames),HEIGHT, WIDTH, 3])
    labels = []
    for i in range(len(filenames)) :
        
        for m in range(len(object_types)):
            if filenames[i].find(object_types[m]) != -1 :
                labels.append(m)           
        
    image = scipy.misc.imread(filenames[i])
    dim = (WIDTH, HEIGHT)
    image = cv2.resize(image, dim)
    images[i] = image
    labels =  np_utils.to_categorical(labels)

    return (images, labels)

(train_images, train_labels) = read_data(train_data_dir);
(test_images, test_labels) = read_data(validation_data_dir);


print(train_labels.shape)
print(train_labels[0])

#create model
model = Sequential()
model.add(Conv2D(32, (4,4),activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (4,4),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (4,4),activation='relu',))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten()) 
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(object_types),activation='softmax'))

model.compile(loss='categorical_crossentropy',
	optimizer='adam',
	metrics=['accuracy'])


#train & test data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range = 0.2,
	width_shift_range = 0.15,
	height_shift_range = 0.15,
	rotation_range = 10,
    horizontal_flip = True)

train_generator = train_datagen.flow(train_images, train_labels)


# trains 
history = model.fit_generator(
    train_generator,
    validation_data = (test_images, test_labels),
    verbose = 1,
    steps_per_epoch = 500,
    epochs = 4)


#plot
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#save model
model_json = model.to_json()
with open("model_demo.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("weights_demo.h5")

