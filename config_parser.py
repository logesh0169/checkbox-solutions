import configparser

config = configparser.ConfigParser()

config['PARENT_DIR'] = "/home/logeshbabu/workspace/checkbox-vqa/"


with open('pipeline.ini', 'w') as configfile:
    config.write(configfile)