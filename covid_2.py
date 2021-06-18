import numpy as np
from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

#Define Directories for train, test & Validation Set
train_path = 'dataset/train'
test_path = 'dataset/test'
valid_path = 'dataset/val'

batch_size = 10

img_height = 224
img_width = 224

test_data_gen = ImageDataGenerator(rescale = 1./255)
image_gen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

def train():

    train = image_gen.flow_from_directory(train_path, target_size=(img_height, img_width), color_mode='rgb', batch_size=batch_size)
    valid = test_data_gen.flow_from_directory(valid_path, target_size=(img_height, img_width), color_mode='rgb', batch_size=batch_size)

    vgg = VGG_model()

    vgg16_features = vgg.predict(train)

    steps = [('pca', PCA(n_components=15)), ('m', LogisticRegression())]

    model = Pipeline(steps=steps)
    model.fit(vgg16_features)

    cnn = create_model()

    early = EarlyStopping(monitor='val_loss', mode='min', patience=2)
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.3, min_lr=0.000001)

    callbacks_list = [early, learning_rate_reduction]

    weights = compute_class_weight('balanced', np.unique(train.classes), train.classes)
    cw = dict(zip( np.unique(train.classes), weights))

    cnn.fit(pca_train, train.classes, epochs=25, validation_data=valid, class_weight=cw, callbacks=callbacks_list)

    cnn.save_weights("covid.h5")

def test():

    test = test_data_gen.flow_from_directory(test_path, target_size=(img_height, img_width), color_mode='grayscale', batch_size=batch_size, shuffle=False)

    cnn = create_model()

    cnn.load_weights("covid.h5")

    test_accu = cnn.evaluate(test)
    print('The testing accuracy is :', test_accu[1]*100, '%')

    preds = np.argmax(cnn.predict(test), axis=-1)

    print(preds)

    cm = pd.DataFrame(data=confusion_matrix(test.classes, preds), index=["Actual Normal", "Actual Covid", "Actual Pneumonia"], columns=["Predicted Normal", "Predicted Covid", "Predicted Pneumonia"])
    sns.heatmap(cm, annot=True, fmt="d")

    plt.show()

def create_model():

    cnn = Sequential()

    cnn.add(Conv2D(32, (3, 3), activation="relu", input_shape=(img_width, img_height, 3)))
    cnn.add(MaxPooling2D(pool_size = (2, 2)))
    cnn.add(Conv2D(32, (3, 3), activation="relu", input_shape=(img_width, img_height, 3)))
    cnn.add(MaxPooling2D(pool_size = (2, 2)))
    cnn.add(Conv2D(32, (3, 3), activation="relu", input_shape=(img_width, img_height, 3)))
    cnn.add(MaxPooling2D(pool_size = (2, 2)))
    cnn.add(Conv2D(64, (3, 3), activation="relu", input_shape=(img_width, img_height, 3)))
    cnn.add(MaxPooling2D(pool_size = (2, 2)))
    cnn.add(Conv2D(64, (3, 3), activation="relu", input_shape=(img_width, img_height, 3)))
    cnn.add(MaxPooling2D(pool_size = (2, 2)))
    cnn.add(Flatten())
    cnn.add(Dense(units = 128, activation = 'relu'))
    cnn.add(Dense(units = 64, activation = 'relu'))
    cnn.add(Dense(units = 3, activation = 'softmax'))
    cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    return cnn

def VGG_model():

    vgg = VGG16(weights='imagenet', include_top=False)

    return vgg

if __name__ == '__main__':
    train()
    test()