import numpy as np
from PIL import Image
import glob
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


#Define Directories for train, test & Validation Set
train_path = 'dataset/train'
test_path = 'dataset/test'
valid_path = 'dataset/val'

batch_size = 10

img_height = 299
img_width = 299

def main():

    image_gen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
    test_data_gen = ImageDataGenerator(rescale = 1./255)

    train = image_gen.flow_from_directory(train_path, target_size=(img_height, img_width), color_mode='grayscale', class_mode='binary', batch_size=batch_size)
    test = test_data_gen.flow_from_directory(test_path, target_size=(img_height, img_width), color_mode='grayscale', shuffle=False, class_mode='binary', batch_size=batch_size)
    valid = test_data_gen.flow_from_directory(valid_path, target_size=(img_height, img_width), color_mode='grayscale', class_mode='binary', batch_size=batch_size)

    cnn = create_model()

    early = EarlyStopping(monitor='val_loss', mode='min', patience=3)
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.3, min_lr=0.000001)

    callbacks_list = [ early, learning_rate_reduction]

    weights = compute_class_weight('balanced', np.unique(train.classes), train.classes)
    cw = dict(zip( np.unique(train.classes), weights))

    cnn.fit(train, epochs=25, validation_data=valid, class_weight=cw, callbacks=callbacks_list)

    test_accu = cnn.evaluate(test)
    print('The testing accuracy is :', test_accu[1]*100, '%')

    preds = cnn.predict(test, verbose=1)

    predictions = preds.copy()
    predictions[predictions <= 0.5] = 0
    predictions[predictions > 0.5] = 1

    cm = pd.DataFrame(data=confusion_matrix(test.classes, predictions, labels=[0, 1]),index=["Actual Normal", "Actual Covid"],
    columns=["Predicted Normal", "Predicted Covid"])
    sns.heatmap(cm, annot=True, fmt="d")

    plt.show()

def create_model():

    cnn = Sequential()

    cnn.add(Conv2D(32, (3, 3), activation="relu", input_shape=(img_width, img_height, 1)))
    cnn.add(MaxPooling2D(pool_size = (2, 2)))
    cnn.add(Conv2D(32, (3, 3), activation="relu", input_shape=(img_width, img_height, 1)))
    cnn.add(MaxPooling2D(pool_size = (2, 2)))
    cnn.add(Conv2D(32, (3, 3), activation="relu", input_shape=(img_width, img_height, 1)))
    cnn.add(MaxPooling2D(pool_size = (2, 2)))
    cnn.add(Conv2D(64, (3, 3), activation="relu", input_shape=(img_width, img_height, 1)))
    cnn.add(MaxPooling2D(pool_size = (2, 2)))
    cnn.add(Conv2D(64, (3, 3), activation="relu", input_shape=(img_width, img_height, 1)))
    cnn.add(MaxPooling2D(pool_size = (2, 2)))
    cnn.add(Flatten())
    cnn.add(Dense(activation = 'relu', units = 128))
    cnn.add(Dense(activation = 'relu', units = 64))
    cnn.add(Dense(activation = 'sigmoid', units = 1))
    cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    return cnn

if __name__ == '__main__':
    main()
    