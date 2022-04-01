import requests, os, sys
import pprint
import json
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import matplotlib.pyplot as plt
import pandas as pd
sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util

__MODEL_KEYS__ = {'ingredients': 
                    {'model_id': 'b80ebaba-f790-4b20-912c-6e5f4f5c1cac', 
                    'api_key': '3CAHKciTzzvTkrbwhx01RYxbBq4aLDkz', 
                    'path_to_label_map': 'gs://wizyvision-automl/wizyocr/ingredients/ingredients_label_map.pbtxt'
                    },
                  'digits': 
                    {'model_id': '5e548aea-34e1-4b52-b90f-338023830cb1', 
                    'api_key': '3CAHKciTzzvTkrbwhx01RYxbBq4aLDkz', 
                    'path_to_label_map': 'gs://wizyvision-automl/wizyocr/digits/label_map.pbtxt'
                    }
}

class WizyOCR(object):
    def __init__(self, name='IngredientsOCR', num_classes=1):
        self.name = name
        self.label_map = label_map_util.load_labelmap(__MODEL_KEYS__[name]['path_to_label_map'])
        self.num_classes = num_classes
        self.model_id = __MODEL_KEYS__[name]['model_id'] 
        self.api_key = __MODEL_KEYS__[name]['api_key']
        self.url = 'https://app.nanonets.com/api/v2/ObjectDetection/Model/' + self.model_id + '/LabelFile/'
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=self.num_classes, use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)

    def get_image(self, image_path):
        self.image_path = image_path
        
#     def upload_image:
    
    
    def get_response(self):
        data = {'file': open(self.image_path, 'rb'),    'modelId': ('', self.model_id)}

        response = requests.post(self.url, auth=requests.auth.HTTPBasicAuth(self.api_key, ''), files=data)

        # print(response.text)

        self.json_data = json.loads(response.text)
        self.predictions = self.json_data['result'][0]['prediction']
        self.xmin = self.predictions[0]['xmin']
        self.ymin = self.predictions[0]['ymin']
        self.xmax = self.predictions[0]['xmax']
        self.ymax = self.predictions[0]['ymax']
        self.ocr_text = self.predictions[0]['ocr_text']

    def get_detection(self):
        self.image = cv2.imread(self.image_path)
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.image_expanded = np.expand_dims(self.image_rgb, axis=0)
        self.image_shape = self.image.shape

        self.df_boxes = [[box['ymin']/self.image_shape[0], box['xmin']/self.image_shape[1], box['ymax']/self.image_shape[0], box['xmax']/self.image_shape[1]] for box in self.json_data['result'][0]['prediction']]
        self.df_scores = [box['score'] for box in self.json_data['result'][0]['prediction']]
        self.df_labels = [box['label'] for box in self.json_data['result'][0]['prediction']]
        self.df_labels = [1 for df_label in self.df_labels]
        self.df_ocr_texts = [box['ocr_text'] for box in self.json_data['result'][0]['prediction']]

        self.correct_boxes = np.array([self.df_boxes], dtype=np.float32)
        self.correct_classes = np.array([self.df_labels], dtype=np.float32)

        self.actual_boxes = np.array([self.df_boxes + [[0.0, 0.0, 0.0, 0.0]]*(100-len(self.df_boxes))], dtype=np.float32)
        self.correct_scores = np.array([[1.0]*len(self.correct_boxes[0])], dtype=np.float32)

        self.actual_classes = np.array([self.df_labels + [10.]*(100-len(self.df_labels))], dtype=np.float32)
        self.actual_scores = np.array([self.df_scores + [0.0]*(100-len(self.actual_boxes))], dtype=np.float32)

    def predict(self, image_path):
        self.get_image(image_path)
        self.get_response()
        self.get_detection()
        vis_util.visualize_boxes_and_labels_on_image_array(
            self.image,
            np.squeeze(self.actual_boxes),
            np.squeeze(self.actual_classes).astype(np.int32),
            np.squeeze(self.actual_scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.60)

        fig, ax = plt.subplots(1,1, figsize=(20,20))
        plt.figure(figsize = (20,20))
        ax.imshow(self.image)
        plt.show()
        
        print("OCR Output: ")
        for text in self.df_ocr_texts:
            print(text + "\n")