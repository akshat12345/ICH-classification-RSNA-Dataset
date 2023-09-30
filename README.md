# ICH-classification-RSNA-Dataset
This repository contains some 3D CNN models that were trained in ICH classification on RSNA Dataset. ICH Classification on RSNA Dataset was a challenge that was hosted on Kaggle(https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/rsna-intracranial-hemorrhage-detection-challenge-2019).

File Description - 
* preprocess_data.py - This file has the code for preprocessing the patient's data in the dataset. It loads all '.dicom' files for patients and converts them into numpy array. Then numpy arrays are clubbed together with a depth of 3 or 8 to get a 3D slice.
* 3D_CNN - This file contains a very simplistic 3D CNN model.
* Keras_Resnet3D - This file contains a 3D version of Resnet model.
* Block-Based 3D CNN - A model in which the whole structure is defined in terms of blocks. Implementation of this research paper https://www.sciencedirect.com/science/article/pii/S1746809422002427.

Keywords

ICH - Intracranial Hemorrhage, RSNA - Radiological Society of North America, CNN - Convolutional Neural Network
