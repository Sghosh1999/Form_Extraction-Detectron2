import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2 import model_zoo

from detectron2.utils.visualizer import ColorMode 

import random
import cv2
import matplotlib.pyplot as plt 
import numpy as np
from PIL import Image
from matplotlib.patches import Rectangle
import pytesseract
import easyocr


import re

reader = easyocr.Reader(["en"], gpu=False)

def plot_samples(dataset_name, n=1):
    dataset_custom = DatasetCatalog.get(dataset_name)
    dataset_custom_metadata = MetadataCatalog.get(dataset_name)
    for s in random.sample(dataset_custom, n):
        img = cv2.imread(s["file_name"])
        v = Visualizer(img[:, :, ::-1], metadata=dataset_custom_metadata, scale=0.5)
        v = v.draw_dataset_dict(s)
        plt.figure(figsize=(15,20))
        plt.imshow(v.get_image())
        plt.show()
        cropped_images_path = os.path.join(os.getcwd(), 'cropped_images', s["file_name"])
        print(cropped_images_path)
        plt.savefig()

def get_train_cfg(config_file_path,
                  checkpoint_url,
                  train_dataset_name,
                  test_dataset_name,
                  num_classes, device, output_dir):
    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file(config_file_path))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint_url)
    cfg.DATASETS.TRAIN = (train_dataset_name, )
    cfg.DATASETS.TEST =  (test_dataset_name, )

    cfg.DATALOADER.NUM_WORKERS = 2

    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 1000
    cfg.SOLVER.STEPS = []
    
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.DEVICE = device
    cfg.OUTPUT_DIR = output_dir
    return cfg

def on_image(image_path, predictor):
    im = cv2.imread(image_path)
    outputs = predictor(im)
    vobj = Visualizer(im[:,:,::-1], metadata={}, scale=0.5, instance_mode=ColorMode.SEGMENTATION)
    vobj = vobj.draw_instance_predictions(outputs["instances"].to("cpu"))

    plt.figure(figsize=(15,20)) ## Show masked image
    plt.imshow(vobj.get_image()) 
    plt.show()
    return 

import os

def crop_image(image_path, predictor):
    im = cv2.imread(image_path)
    outputs = predictor(im)

    boxes_list = outputs["instances"].pred_boxes.tensor.cpu().numpy()
    scores_list = outputs["instances"].scores.detach().cpu().numpy()
    pred_classes_list = outputs["instances"].pred_classes.detach().cpu().numpy()


    #print(pred_classes_list)

    cropped_images_dict = {}

    reader = easyocr.Reader(['en'])

    # # Create the "cropped_images" folder if it doesn't exist
    # output_folder = os.path.join(os.getcwd(), 'cropped_images')
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)

    for i, (box, score, pred_cls) in enumerate(zip(boxes_list, scores_list, pred_classes_list), start=1):
        x_min, y_min, x_max, y_max = map(int, box)
        cropped_image = im[y_min:y_max, x_min:x_max, :]

        # Use EasyOCR to extract text
        results = reader.readtext(cropped_image)

        # Extract text and concatenate into a single string
        concatenated_text = ' '.join([item[1] for item in results])
        

        cropped_images_dict[pred_cls] = concatenated_text

        # # Save the cropped image
        # cropped_file_name = f"crop_{os.path.splitext(os.path.basename(image_path))[0]}_{i}.jpg"
        # cropped_file_path = os.path.join(output_folder, cropped_file_name)
        # cv2.imwrite(cropped_file_path, cropped_image)

        # Display the original image with bounding boxes
        # plt.figure(figsize=(8, 8))
        # plt.imshow(im)
        # current_axis = plt.gca()
        # #print(results)
        # rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
        #                  linewidth=2, edgecolor='r', facecolor='none')
        # current_axis.add_patch(rect)
        # plt.title(f"Class: {pred_cls}, Score: {score}")
        # plt.show()

    return cropped_images_dict

    

# def extract_text_cropped_images(image_path_dict):

#     print('======================================================================================', end='\n\n')
#     for pred_cls, image_path in image_path_dict.items():
#         cv_image = cv2.imread(image_path)
#         result = reader.readtext(cv_image, detail=0, paragraph=False)
#         para_res = ' '.join(result).upper()
#         # print('IMAGE: ', [str(img_idx)+'.jpg'])
#         print('RESULT DATA: ',  para_res)




























































        
        # invoice_no = re.search(r'Invoice No (\d+)', para_res, flags=re.I)
        # if invoice_no:
        #     invoice_no = invoice_no.groups()[0]
        #     # print('INVOICE NO:', [invoice_no])
        
        # cash_amount = re.search(r'Cash ([0-9.,]+)', para_res, flags=re.I)
        # if cash_amount:
        #     cash_amount = cash_amount.groups()[0]
        #     # print('CASH:', [cash_amount])
        
        # purchase_amount = re.search(r'PURCHASE AMOUNT.* [A-Z]{3} ([0-9.,]+)', para_res, flags=re.I)
        # if purchase_amount:
        #     purchase_amount = purchase_amount.groups()[0]
        #     # print('PURCHASE AMT:', [purchase_amount])
        
        # approval_code = re.search(r'APPROVAL CODE \'?([0-9.,]+)', para_res, flags=re.I)
        # if approval_code:
        #     approval_code = approval_code.groups()[0]
        #     # print('APPROVAL CODE:', [approval_code])
    print('========================================================================================')
    





    
    








