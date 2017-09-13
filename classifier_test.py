from keras.models import model_from_json  
import numpy as np
import glob
import scipy.misc
import cv2

test_data_dir = './test_data/*.jpeg'

WIDTH = 100
HEIGHT = 55



#oad model
json_file = open('model_demo.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("weights_demo.h5")

#object types
object_types = []

filenames = glob.glob(test_data_dir)
for i in range(len(filenames)) :
    image = scipy.misc.imread(filenames[i])
    dim = (WIDTH, HEIGHT)
    image = cv2.resize(image, dim)
    image = np.reshape(image, (1, HEIGHT, WIDTH, 3))
    prediction = loaded_model.predict(image)
    print(prediction)
    label = np.argmax(prediction)
    card_type = object_types[label]
    print("File " + filenames[i] + ": " + card_type)
