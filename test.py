import os
import time
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

start = time.time()
from detectron2.engine import DefaultPredictor
import pickle

from utils import *
import pytesseract
import os
import cv2
import re
import matplotlib.pyplot as plt
import pytesseract
from fuzzywuzzy import fuzz

pytesseract.pytesseract.tesseract_cmd = r'C:/Users/MP158YC/AppData/Local/Programs/Tesseract-OCR/tesseract.exe'



cfg_save_path = "t4a_cfg.pickle"

with open(cfg_save_path, 'rb') as f:
    cfg = pickle.load(f)

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8

predictor = DefaultPredictor(cfg)

mapping_dict = {0: "Social Insurance Number",
 1: "receipents program account number",
 2: "Pension or Superannuation-line 11500",
 3: "lum-sum amount-line 13000",
 4: "self-employed commisions",
 5: "income-tax deducted-line 43700",
 6: "Annuities renties",
 7: "fees for services",
 8: "payers program account number",
 9: "box_case amount montant_1",
 10: "box_case amount montant_10",
 11: "box_case amount montant_11",
 12: "box_case amount montant_12",
 13: "box_case amount montant_2",
 14: "box_case amount montant_3",
 15: "box_case amount montant_4",
 16: "box_case amount montant_5",
 17: "box_case amount montant_6",
 18: "box_case amount montant_7",
 19: "box_case amount montant_8",
 20: "box_case amount montant_9",
 21: "payersname",
 22: "recipients name and address",
 23: "Year"}


# Example usage
image_path = "test/t4a_39_augmented.PNG"
cropped_images_dict = crop_image(image_path, predictor)


# Create a new dictionary by mapping values from mapping_dict to keys of text_dict
final_mapping = {mapping_dict[key]: value for key, value in cropped_images_dict.items()}
final_mapping_modified = {}

# Function to remove fuzzy matched substring
def remove_fuzzy_substring(original_str, substring):
    ratio = fuzz.token_set_ratio(original_str, substring)
    if ratio > 50:
        return original_str.replace(substring, '')
    return original_str

def final_json_creation(final_mapping):
    for key, value in final_mapping.items():
        if key in ['payersname']:
            # Keep the original value for specified keys
            final_mapping_modified[key] = value
        elif key in ['recipients name and address']:
            # Remove fuzzy matched substring for the specified key
            value_cleaned = remove_fuzzy_substring(value, 'Recipients name and address Nom et adresse du beneficiaire Laslname Iprin ) amine Whme  IT Wuls IFmanate preram Inuie nba')
            final_mapping_modified[key] = value_cleaned.strip()
        else:
            # Remove non-integer characters, excluding "/", and limit consecutive spaces to three
            value_cleaned = re.sub(r'(?<!/)\s{3,}', '', ''.join(char if char.isdigit() or char == '/' or char.isspace() else '' for char in value))
            if key in ['payers program account number']:
                final_mapping_modified[key] = value_cleaned[3:]
            else:
                final_mapping_modified[key] = value_cleaned
        return final_mapping_modified

print(final_mapping_modified)
