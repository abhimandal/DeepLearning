# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
# Hidden Layer nodes to be not small and better to be power of 2
classifier.add(Dense(units = 128, activation = 'relu')) 

# binary outcome so we use sigmoid activation instead of softmax
classifier.add(Dense(units = 1, activation = 'sigmoid')) 

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.summary()

# Part 2 - Fitting the CNN to the images
# Image augmentation to avoid overfitting 

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# From https://keras.io/preprocessing/image/
# Trying multiple random image transformations in different batches
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


# fit classifier on train and simultaneously test performace  on test set
#steps corresponds to the original quantity of images in respective sets
classifier.fit_generator(training_set,
                         steps_per_epoch = 8000 // 32,
                         epochs = 20,
                         validation_data = test_set,
                         validation_steps = 2000 // 32)

classifier.save_weights('./checkpoints/2ConvLayer')
classifier.save('2ConvLayer_model.h5')


# Part 3 - Making new predictions

import numpy as np
from tensorflow.keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image) # to create 3D (for RGB)

# to make batches for predict fn corresponding to batch 
test_image = np.expand_dims(test_image, axis = 0) 

# predict the new image
result = classifier.predict(test_image)
training_set.class_indices # to see the mapping between class labels and final predictions
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'