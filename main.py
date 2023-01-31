import time
import warnings

from pathlib import Path
from utils.config import *
from utils.file_info import *
from utils.logger import logger

from src.detection.checkbox_detection_test import CheckboxDetectionTest
from src.detection.checkbox_attr_detection_test import CheckboxAttrDetectionTest
from src.recognition.checkbox_state_recognition_test import CheckboxStateRecognition
from src.extraction.printed_text_extraction import PrintedTextExtraction

warnings.filterwarnings('ignore')


# main program
if __name__ == '__main__':

    logger.info("CHECKBOX VQA STARTED")

    start = time.time()

    res = {}
    checkbox_status = []

    file_name = Path(INPUT_FILE).stem
    file_extension = Path(INPUT_FILE).suffix
    file_size = get_file_size(INPUT_FILE)
    file_checksum = get_checksum(INPUT_FILE)

    logger.info("CHECKBOX ATTRIBUTION DETECTION STARTED")

    checkbox_attr_detection = CheckboxAttrDetectionTest(INPUT_FILE, SOURCE_OF_TRUTH_DIR, CAD_MODEL_PATH)
    checkbox_attr_detected_path = checkbox_attr_detection.checkbox_attr_detection()

    logger.info("CHECKBOX ATTRIBUTION DETECTION END")

    res['inputFilePath'] = INPUT_FILE
    res['inputFileName'] = file_name
    res['fileExtension'] = file_extension
    res['inputFileSize'] = file_size
    res['inputFileChecksum'] = file_checksum

    if not os.listdir(checkbox_attr_detected_path):
        logger.info("Checkbox Attributes Not Found")

        print(res)
    else:

        logger.info("CHECKBOX RESULTS STARTED")

        for checkbox_attr_detected_file in os.listdir(checkbox_attr_detected_path):

            checkbox_attr_detected_file = os.path.join(checkbox_attr_detected_path, checkbox_attr_detected_file)

            checkbox_detection = CheckboxDetectionTest(checkbox_attr_detected_file, SOURCE_OF_TRUTH_DIR, CD_MODEL_PATH)
            detected_file = checkbox_detection.checkbox_detection()

            checkbox_state_recognition = CheckboxStateRecognition(detected_file, CR_MODEL_PATH, CR_IMG_WIDTH,
                                                                  CR_IMG_HEIGHT)
            checkbox_state = checkbox_state_recognition.checkbox_state_recognition()

            printed_text_extraction = PrintedTextExtraction(checkbox_attr_detected_file, PRINTED_TEXT_DETECTION_SMALL_MODEL)
            extracted_printed_text = printed_text_extraction.printed_text_extraction()

            checkbox_status.append({checkbox_state: extracted_printed_text})

            continue

        logger.info("CHECKBOX RESULTS END")

        res['checkbox_status'] = checkbox_status
        end = time.time()

        print(res)

    logger.info("CHECKBOX VQA END")
