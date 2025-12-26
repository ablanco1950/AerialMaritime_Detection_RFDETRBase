# AerialMaritime_Detection_RFDETRBase
Test for applying object detection in maritime aerial views with RFDETRBase, using https://universe.roboflow.com/jacob-solawetz/aerial-maritime/dataset/24 ​​as the custom dataset.

Installation:

Download and extract the project to a folder.

Download the custom dataset file from https://universe.roboflow.com/jacob-solawetz/aerial-maritime/dataset/24 ​​using the COCO JSON format option (a Roboflow user key is required, but this can be obtained for free from the Roboflow website and provides access to information and other features). Verify that it downloads to a folder named Aerial Maritime.v24i.coco with three subfolders: train, valid, and test. Copy this folder to the project folder.

The following module is required, or you must install it:

pip install rfdetr

Model training with a custom dataset

python TRAIN_AerialMaritime_Detection_RFDETRBase.py

In the output folder, .pth models and other log files are created.

For evaluation, object detection is limited to jet skis only, as otherwise an excessive number of boxes could be generated, hindering proper evaluation. Due to their large size, it was not possible to upload the .pth files to GitHub.

python TEST_AerialMaritime_JetSki_RFDETRBase.py

Observations:

The test was performed by training the model on a personal computer that was also used for normal tasks during training: other applications, internet browsing, etc. RFDETRBase consumes a lot of resources during training, so despite choosing a custom dataset with few records, the 20 epochs considered necessary could never be executed. The training ended with a message that a new instance of python.exe could not be run. Two screenshots of the evaluation are included, showing a very low threshold: 0.3. Its accuracy is disregarded because the model evaluation found at https://universe.roboflow.com/jacob-solawetz/aerial-maritime/dataset/24 ​​is confusing.

References:

https://blog.roboflow.com/train-rf-detr-on-a-custom-dataset/

https://universe.roboflow.com/jacob-solawetz/aerial-maritime/dataset/24

A similar project, but using YOLO:

https://github.com/ablanco1950/Radar_Marine-Yolov11

![Fig1](https://github.com/ablanco1950/AerialMaritime_Detection_RFDETRBase/blob/main/jetski3.png)

![Fig1](https://github.com/ablanco1950/AerialMaritime_Detection_RFDETRBase/blob/main/jetski5.png)
