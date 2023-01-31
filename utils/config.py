import os

# parent directory
PARENT_DIR = "/home/logeshbabu/workspace/checkbox-vqa/"

# output directory
OUTPUT_DIR = "../output/"
OUTPUT_DIR = os.path.join(PARENT_DIR, OUTPUT_DIR)

# source of truth
SOURCE_OF_TRUTH_DIR = "source_of_truth/"
SOURCE_OF_TRUTH_DIR = os.path.join(OUTPUT_DIR, SOURCE_OF_TRUTH_DIR)

# system of records
SYSTEM_OF_RECORDS_DIR = "system_of_records/"
SYSTEM_OF_RECORDS_DIR = os.path.join(OUTPUT_DIR, SYSTEM_OF_RECORDS_DIR)

# input file directory

SOURCE_OF_ORIGIN = "/home/logeshbabu/H Hart Packet/H Hart Packet-18.jpg"

INPUT_FILE = os.path.join(PARENT_DIR, SOURCE_OF_ORIGIN)

# checkbox attr detection hyper_parameters

CAD_MODEL_PATH = os.path.join(PARENT_DIR, "/data/loki-workspace/checkbox-vqa/model/v5.0/checkbox_attr_detection_v5.0.pth")

# checkbox detection hyper_parameters

CD_MODEL_PATH = os.path.join(PARENT_DIR, "/data/loki-workspace/checkbox-vqa/model/v3.0/checkbox_detection_v3.0.pth")

# checkbox state recognition hyper_parameters

CR_IMG_WIDTH = 50
CR_IMG_HEIGHT = 50

CR_MODEL_PATH = os.path.join(PARENT_DIR, "/data/loki-workspace/checkbox-vqa/model/checkbox_state_recognition_model_v5.1.0.h5")

# text extraction hyper_parameters

PRINTED_TEXT_DETECTION_SMALL_MODEL = "microsoft/trocr-small-printed"
PRINTED_TEXT_DETECTION_LARGE_MODEL = "microsoft/trocr-large-printed"

