import os
import cv2
import warnings
import torch
import torchvision
import matplotlib.pyplot as plt

from pathlib import Path

warnings.filterwarnings('ignore')


class CheckboxAttrDetectionTest:
    # The init method or constructor
    def __init__(self, input_file, output_dir, model_path):
        # Instance Variable
        self.input_file = input_file
        self.output_dir = output_dir
        self.model_path = model_path

    """  checkbox attribution detection """

    def checkbox_attr_detection(self):

        images = None

        detection_path = "checkbox_attr_detection"
        detection_path = os.path.join(self.output_dir, detection_path)
        os.makedirs(detection_path, exist_ok=True)

        # get a file name and file extension
        file_name = Path(self.input_file).stem
        file_extension = Path(self.input_file).suffix

        detection_path = os.path.join(detection_path, file_name)
        os.makedirs(detection_path, exist_ok=True)

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        model = torch.load(self.model_path)
        model.eval()

        input_image = cv2.imread(self.input_file)
        input_img = torchvision.transforms.functional.to_tensor(input_image).to(device)

        checkbox_tensors = model([input_img])[0]["boxes"]
        prediction = checkbox_tensors.data.cpu().numpy()

        in_img = cv2.imread(self.input_file)

        count = 0
        for checkbox_tensor in prediction:
            checkbox_tensor = [int(i) for i in checkbox_tensor]
            detected_file = f'{detection_path}/{file_name}_{count}.jpg'
            cv2.imwrite(detected_file, input_image[checkbox_tensor[1]:checkbox_tensor[3], checkbox_tensor[0]:checkbox_tensor[2]])

            images = cv2.rectangle(in_img, (checkbox_tensor[0], checkbox_tensor[1]), (checkbox_tensor[2], checkbox_tensor[3]),
                                   (255, 0, 0), 2)
            count += 1

        plt.figure(figsize=(15, 8))
        plt.imshow(images, cmap='gray')
        plt.title('Checkbox Attribution Detection')
        plt.show()

        return detection_path


