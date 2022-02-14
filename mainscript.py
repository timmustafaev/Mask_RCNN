#TRAINING SCRIPT

from google.colab import drive
drive.mount('/content/drive/')

!git clone https://github.com/akTwelve/Mask_RCNN 
  
%cd Mask_RCNN/
%ls

!pip3 install -r requirements.txt

import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import json
import imgaug

%cd /content/Mask_RCNN/

ROOT_DIR = os.path.abspath(
    "/content/drive/MyDrive/TFG/Mask_RCNN/")
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(
    ROOT_DIR, "mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
   
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession  

DATA_DIR = "/content/drive/MyDrive/TFG/FOTOS/LabelMe_JSON"
DATASET_TRAIN_DIR = os.path.join(DATA_DIR, "train")

DATASET_VAL_DIR = os.path.join(DATA_DIR, "val")
DATASET_TEST_DIR = os.path.join(DATA_DIR, "test")

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

class PotatoConfig(Config):
  NAME = "potatos"
  IMAGES_PER_GPU = 1
  NUM_CLASSES = 1 + 5
  STEPS_PER_EPOCH = 70
  DETECTION_MIN_CONFIDENCE = 0.9
  USE_MINI_MASK = False
  IMAGE_SHAPE = [1024,1024,3]
  LEARNING_RATE = 0.0005

config = PotatoConfig()
config.display()

from PIL import Image,ImageDraw
axis_Width = 15 

class PotatoDataset(utils.Dataset):
  def load_dataset(self,dataset_dir):
    self.add_class('potatos', 1, 'pat1')
    self.add_class('potatos', 2, 'pat2')
    self.add_class('potatos', 3, 'pat3')
    self.add_class('potatos', 4, 'pat4')
    self.add_class('potatos', 5, 'pat5')
    for i, filename in enumerate(os.listdir(dataset_dir)):
        annotation_file = os.path.join(dataset_dir, 
                                       filename.replace('.jpg','.json'))
        if '.jpg' in filename and os.path.isfile(annotation_file):
                self.add_image('potatos', 
                               image_id=i,
                               path=os.path.join(dataset_dir, filename),
                               annotation=annotation_file)
    
  def extract_masks(self,filename):
    json_file = os.path.join(filename)
    with open(json_file) as f:
        img_anns = json.load(f)
    n_masks = 0
    for anno in img_anns['shapes']:
        if anno['label']=='pat1' or anno['label']=='pat2' or anno['label']=='pat3' or anno['label']=='pat4' or anno['label']=='pat5':
            n_masks+=1
            
    masks = np.zeros([img_anns['imageHeight'], 
                      img_anns['imageWidth'], n_masks], 
                     dtype='uint8')
    classes = []
    i=0
    for anno in img_anns['shapes']:
        if anno['label']=='pat1' or anno['label']=='pat2' or anno['label']=='pat3' or anno['label']=='pat4' or anno['label']=='pat5':
           if anno['shape_type']=='polygon':
              mask = np.zeros([img_anns['imageHeight'], 
                               img_anns['imageWidth']], dtype=np.uint8) 
              cv2.fillPoly(mask, np.array([anno['points']], 
                                          dtype=np.int32), 1)
              masks[:, :, i] = mask                                    
              classes.append(self.class_names.index(anno['label']))
              i+=1
    return masks, classes
  def load_mask(self, image_id):
      info = self.image_info[image_id]
      path = info['annotation']
      masks, classes = self.extract_masks(path)
      return masks, np.asarray(classes, dtype='int32')
    
  def image_reference(self, image_id):
      info = self.image_info[image_id]
      return info['path']
    
dataset_train = PotatoDataset()
dataset_train.load_dataset(DATASET_TRAIN_DIR)
dataset_train.prepare()

dataset_val = PotatoDataset()
dataset_val.load_dataset(DATASET_VAL_DIR)
dataset_val.prepare()

dataset_test = PotatoDataset()
dataset_test.load_dataset(DATASET_TEST_DIR)
dataset_test.prepare()

%load_ext tensorboard

model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

init_with = "last"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    model.load_weights(model.find_last(), by_name=True)

%tensorboard --logdir /content/drive/MyDrive/TFG/Mask_RCNN/logs/potatos20211126T2055/

Potato_augmentation = imgaug.augmenters.Sometimes(0.5,
    [imgaug.augmenters.geometric.Affine(rotate=(-360,360))])
    
    
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=150,
            layers='heads',augmentation = Potato_augmentation)
  
#INFERENCE SCRIPT
class InferenceConfig(PotatoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()


model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)


print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

image_id = 13
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
  modellib.load_image_gt(dataset_test, inference_config, 
                           image_id)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                            dataset_train.class_names, figsize=(8, 8))

results = model.detect([original_image], verbose=1)

r = results[0]
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                          dataset_val.class_names, r['scores'], ax=get_ax())#foto
print(r['class_ids'].shape)
print(r["class_ids"])

#CONFUSION MATRIX SCRIPT
import pandas as pd
import numpy as np
import os 
!git clone https://github.com/Altimis/Confusion-matrix-for-Mask-R-CNN

from confusion import utils as ut
gt_tot = np.array([])
pred_tot = np.array([])
#mAP list
mAP_ = []

for image_id in dataset_test.image_ids:
  image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset_test, config, image_id)
  info = dataset_test.image_info[image_id]
  results = model.detect([image],verbose=1)
  r = results[0]
  gt, pred = ut.gt_pred_lists(gt_class_id, gt_bbox, r['class_ids'], r['rois'])
  gt_tot = np.append(gt_tot, gt)
  pred_tot = np.append(pred_tot, pred)
  AP_, precision_, recall_, overlap_ = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                          r['rois'], r['class_ids'], r['scores'], r['masks'])
  mAP_.append(AP_)
  
gt_tot=gt_tot.astype(int)
pred_tot=pred_tot.astype(int)
save_dir = "output"
gt_pred_tot_json = {"gt_tot" : gt_tot, "pred_tot" : pred_tot}
df = pd.DataFrame(gt_pred_tot_json)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
df.to_json(os.path.join(save_dir,"gt_pred_test.json"))
tp,fp,fn=ut.plot_confusion_matrix_from_data(gt_tot,pred_tot,fz=18, figsize=(20,20), lw=0.5)
