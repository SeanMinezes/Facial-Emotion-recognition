# Project Summary

## PROJECT TITLE: Facial Emotion Detection

### GOAL

Multiclass classification of facial emotions from grayscale images.

### DATASET

[FER-DS - Kaggle](https://www.kaggle.com/datasets/mhantor/facial-expression)

### DESCRIPTION

* We have 19950 images covering 4 facial emotions (angry, happy, neutral, surprised).
* The images are in grayscale and have a low resolution (48x48).
* Our aim is to use these images to train a deep learning model which can classify them accurately.

### TASKS PERFORMED

1. Data exploration and creation of custom validation and test sets using images sampled from the original dataset.These will be fixed to ensure valid comparison of different models.Final distribution of images: Train - 16159, Validation - 1796, Test - 1995 (Total: 19950)
2. Trained CNN model from scratch as a baseline, trying different configurations for number of layers and hidden units.
3. Added configuration for data-augmentation to the data pipeline, and evaluated their effectiveness using the same CNN architecture from the previous step.

### MODELS IMPLEMENTED

1. Convolutional Neural Network (for baseline model)

(Data-augmentation utilized with this models)

### LIBRARIES NEEDED

1. Tensorflow
2. Keras
3. Keras_CV
4. Numpy
5. Matplotlib
6. Scikit-learn

### VISUALIZATION

1. [Baseline CNN](Images/00_baseline_cnn) ([Notebook 00](Model/00_baseline_cnn.ipynb))
2. [CNN + Data augmentation](Images/01_data_augmentation_cnn) ([Notebook 01](Model/01_data_augmentation_cnn.ipynb)) 

### MODEL PERFORMANCE (BASED ON ACCURACY SCORES)

|    Model configuration    | Val accuracy (%) | Test accuracy (%) |
| :-----------------------: | :--------------: | :---------------: |
|       Baseline CNN       |      71.05      |       71.23       |
| CNN + Simple data-augment |      75.45      |       74.99       |
|                          |                  |                  |

### CONCLUSION

* We used low-resolution, grayscale images to train a deep learning model to detect four facial emotions.
* Starting with a solid baseline using a CNN, we setup a framework for comparison of different model configurations.
* Combining data-augmentation with transfer-learning techniques could improved performance significantly.
* Our final model consists of:
  1. Data augmentation - Rotation + Horizontal flipping
  2. Model - Simple CNN with Augmentation

### SIGNATURE

Sean Minezes


