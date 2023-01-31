import warnings
import easyocr

warnings.filterwarnings('ignore')


class PrintedTextExtraction:

    # The init method or constructor
    def __init__(self, input_file, model):

        # Instance Variable
        self.input_file = input_file
        self.model = model

    """   Printed Text Extraction """

    def printed_text_extraction(self):

        reader = easyocr.Reader(['en'])
        extracted_printed_text = reader.readtext(self.input_file)

        return extracted_printed_text



