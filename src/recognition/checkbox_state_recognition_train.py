import os
import warnings
import matplotlib.pyplot as plt

from keras.layers import Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator

warnings.filterwarnings('ignore')


class CheckboxStateRecognitionTrain:

    # The init method or constructor
    def __init__(self, train_path, test_path, epoch, batch_size, img_width, img_height, model_path, model_name):
        # Instance Variable
        self.train_path = train_path
        self.test_path = test_path
        self.epoch = epoch
        self.batch_size = batch_size
        self.img_width = img_width
        self.img_height = img_height
        self.model_path = model_path
        self.model_name = model_name

    """   Checkbox state Recognition """

    def checkbox_state_recognition(self):

        label = os.listdir(self.train_path)
        num_label = len(label)

        print("Class --> {} \n and the length is : {}".format(label, num_label))

        train_datagen = ImageDataGenerator(
                rescale=1. / 255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True
        )

        training_set = train_datagen.flow_from_directory(
                directory=self.train_path,
                target_size=(self.img_width, self.img_height),
                batch_size=self.batch_size,
                class_mode='categorical'
        )

        test_datagen = ImageDataGenerator(
                rescale=1. / 255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)

        test_set = test_datagen.flow_from_directory(
                directory=self.test_path,
                target_size=(self.img_width, self.img_height),
                batch_size=self.batch_size,
                class_mode='categorical'
        )

        # Import the VGG 16 library as shown below and add preprocessing layer to the front of VGG
        # Here we will be using imagenet weights

        vgg = VGG16(input_shape=[self.img_width, self.img_height] + [3], weights='imagenet', include_top=False)

        # don't train existing weights
        for layer in vgg.layers:
            layer.trainable = False

        # our layers - you can add more if you want
        x = Flatten()(vgg.output)
        prediction = Dense(num_label, activation='softmax')(x)

        # create a model object
        model = Model(inputs=vgg.input, outputs=prediction)
        model.summary()

        # tell the model what cost and optimization method to use
        model.compile(
                loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy']
        )

        # Train the model
        history = model.fit(
                training_set,
                validation_data=test_set,
                epochs=self.epoch,
                steps_per_epoch=len(training_set),
                validation_steps=len(test_set)
        )

        # summarize history for loss
        plt.plot(history.history['loss'], label='Train loss')
        plt.plot(history.history['val_loss'], label='Validation loss')
        plt.title('summarize history for loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        # summarize history for accuracy
        plt.plot(history.history['accuracy'], label='Train acc')
        plt.plot(history.history['val_accuracy'], label='val acc')
        plt.title('summarize history for accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        model_path = self.model_path + self.model_name
        model.save(model_path)

        return model_path

# config


TRAIN_PATH = "/data/loki-workspace/checkbox-vqa/source_of_origin/datasetv5.0/checkbox/recognition/vgg16/train/"
TEST_PATH = "/data/loki-workspace/checkbox-vqa/source_of_origin/datasetv5.0/checkbox/recognition/vgg16/val/"
EPOCH = 100
BATCH_SIZE = 24
IMG_WIDTH = 50
IMG_HEIGHT = 50

MODEL_PATH = "/data/loki-workspace/checkbox-vqa/model/"
MODEL_NAME = "checkbox_state_recognition_model_v5.1.0.h5"

checkbox_state_recognition = CheckboxStateRecognitionTrain(TRAIN_PATH, TEST_PATH, EPOCH, BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, MODEL_PATH, MODEL_NAME)
checkbox_state_recognition.checkbox_state_recognition()


