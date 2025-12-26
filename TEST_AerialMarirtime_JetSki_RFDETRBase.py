# https://blog.roboflow.com/train-rf-detr-on-a-custom-dataset/
# modified by ablanco50, Alfonso Blanco
# dataset  https://universe.roboflow.com/jacob-solawetz/aerial-maritime/dataset/24


THRESHOLD=0.3

# directory of test images
dirname ="Aerial Maritime.v24i.coco\\test"

classes=['jetski']

from rfdetr import RFDETRBase

model = RFDETRBase(pretrain_weights="output\checkpoint.pth")


import supervision as sv
import numpy as np
from PIL import Image


Cont_jetskis_Detected=0
Cont_jetskis_No_Detected=0

import os
import re
def loadimages(dirname):
 
     imgpath = dirname + "\\"      
     
     images = []
     TabFileName=[]   
    
     print("Reading images from ",imgpath)
     NumImage=-2
     
     Cont=0
     for root, dirnames, filenames in os.walk(imgpath):
        
         NumImage=NumImage+1
         
         for filename in filenames:
             
             if re.search("\.(jpg|jpeg|png|bmp|tiff|JPEG)$", filename):
                 
                 
                 filepath = os.path.join(root, filename)
                 image = Image.open(filepath)                
                 
                                         
                 images.append(image)
                 TabFileName.append(filename)
                 
                 Cont+=1
                 
     print("Readed " + str(len(images)))
     
     return images, TabFileName

#
# MAIN
#


images, TabFileName=loadimages(dirname)

for i in range(len(images)):

    image=images[i]

    detections = model.predict(image, threshold=THRESHOLD)

    text_scale = sv.calculate_optimal_text_scale(resolution_wh=image.size)
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=image.size)

    bbox_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
            text_color=sv.Color.BLACK,
            text_scale=text_scale,
            text_thickness=thickness,
            smart_position=True)



    #print(detections.confidence)

    detections_labels = [
            
            f"jetski  {str(confidence)[12:16]}"
            
            for confidence
            
            in zip( detections.confidence)
    ]

    detections_image = image.copy()

    Cont_jetskis=0

    detections_image_jetski=[]
    for i in range(len(detections.class_id)):
      if  detections.class_id[i] == 4:   # class 4  is jetski       
          detections_image_jetski = bbox_annotator.annotate(detections_image, detections[i])
          detections_image_jetski=label_annotator.annotate(detections_image, detections[i], [detections_labels[i]])
          #sv.plot_images_grid(images=[detections_image_jetski], grid_size=(1, 2), titles=["Detection"])
          Cont_jetskis=Cont_jetskis+1

    sv.plot_images_grid(images=[detections_image_jetski], grid_size=(1, 2), titles=["Detection"])
    if Cont_jetskis==0:
         Cont_jetskis_No_Detected= Cont_jetskis_No_Detected +1
    else:
         Cont_jetskis_Detected= Cont_jetskis_Detected + Cont_jetskis

    print("Detected jetskis = " + str(Cont_jetskis))

print("jetskis no detected = " + str(Cont_jetskis_No_Detected))
print("jetskis detected = " + str(Cont_jetskis_Detected))

    
