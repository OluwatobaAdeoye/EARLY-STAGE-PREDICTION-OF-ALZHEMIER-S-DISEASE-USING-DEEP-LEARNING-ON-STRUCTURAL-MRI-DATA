This project leverages structural MRI images alongside deep learning techniques to detect early signs of Alzheimer's Disease. It utilizes Convolutional Neural Networks (CNN), EfficientNetB0, and a hybrid InceptionV3+ResNet50 model for classification.

Dataset 
Source: Kaggle (via CDC)

Total Images: 6,400 structural MRI scans

Classes:

NonDemented

VeryMildDemented

MildDemented

ModerateDemented

Splits 

Train: 80%

Validation: 10%

Test: 10%

Preprocessing 

Unified all data into a common directory

Resized all images to 150x150

Normalized image pixel values

Model Used 

1. Convolutional Neural Network (CNN)
   
3 Conv2D layers with ReLU + MaxPooling

Fully connected Dense layers + Dropout

Test Accuracy: 97.66%

Validation Accuracy: 97.35%

2. EfficientNetB0 (Pretrained)
   
Fine-tuned on the MRI dataset

GlobalAveragePooling + Dense layers

Test Accuracy: 91.43%

Validation Accuracy: 91.56%

3. Hybrid (InceptionV3 + ResNet50)
   
Combined features from both networks

Used Concatenate, BatchNormalization, and Dense layers

Test Accuracy: 90.62%

Validation Accuracy: 92.52%

 Model Comparison
 Metric | Class | CNN | EfficientNetB0 | Hybrid Model
Accuracy | Overall | 98% | 92% | 91%
Precision | NonDemented | 0.98 | 0.94 | 0.92
 | VeryMildDemented | 1.00 | 0.75 | 1.00
 | MildDemented | 0.98 | 0.88 | 0.93
 | ModerateDemented | 0.98 | 0.98 | 0.86
 
Recall | NonDemented | 0.98 | 0.91 | 0.88
 | VeryMildDemented | 0.67 | 1.00 | 0.67
 | MildDemented | 0.98 | 1.00 | 0.92
 | ModerateDemented | 0.98 | 0.80 | 0.90
 
F1 Score | NonDemented | 0.98 | 0.93 | 0.90
 | VeryMildDemented | 0.80 | 0.86 | 0.80
 | MildDemented | 0.98 | 0.93 | 0.93
 | ModerateDemented | 0.98 | 0.88 | 0.88

 Model Comparison (Avg. Precision/Recall/F1)

 Model | Precision | Recall | F1-Score
CNN | 98.50% | 90.25% | 93.50%
EfficientNetB0 | 88.75% | 92.75% | 90.00%
Hybrid Model | 92.75% | 84.25% | 87.75%

Performance Visualization
Confusion matrices and classification reports are included

Training and validation accuracy/loss graphs plotted per epoch

 How to Run
Upload the dataset to your Google Drive under:
/MyDrive/archive (1)/Alzheimer_s Dataset/
Mount Google Drive in Colab

Run the full pipeline: preprocessing, training, and evaluation

Try the interactive prediction demo by running:
user_input_and_accuracy(labels, model)

Notes
Best model (based on F1-score and robustness): CNN

Class imbalance handled during training by combining subtypes into Demented vs NonDemented during visualizations

The notebook contains EDA, visualizations, and model metrics
