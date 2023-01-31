
import keras
import warnings
import numpy as np
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input

warnings.filterwarnings('ignore')


class CheckboxStateRecognition:

    # The init method or constructor
    def __init__(self, input_file, model_path, img_width, img_height):
        # Instance Variable
        self.input_file = input_file
        self.model_path = model_path
        self.img_height = img_height
        self.img_width = img_width

    """   Checkbox state Recognition """
    def checkbox_state_recognition(self):

        img = keras.utils.load_img(self.input_file, target_size=(self.img_width, self.img_height))
        x = keras.utils.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x_processed = preprocess_input(x)

        checkbox_state_model = load_model(self.model_path)
        classes = checkbox_state_model.predict(x_processed)

        cf = ['{:f}'.format(item) for item in classes[0]]

        classes = classes.round()

        if classes[0][0] == 1:

            return 'TICKED-TRUE'

        if classes[0][1] == 1:

            return 'TICKED-FALSE'

        elif classes[0][2] == 1:

            return 'UN-TICKED'

        return None



