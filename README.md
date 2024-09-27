# Emo-EEG-Voice: Multimodal Depression Recognition

This repository contains the code and data used in our study on depression recognition through multimodal emotional feature extraction from EEG and audio data. The study leverages various machine learning and deep learning models to improve the accuracy and robustness of emotion-based depression diagnosis.

## Repository Structure

- **Voice/**  
  This folder contains the Jupyter notebook example used for extracting emotional features from audio data via a **3D CNN-BLSTM model with an attention mechanism**. The notebook demonstrates how to preprocess the audio data, apply the CNN-BLSTM model, and extract emotional features such as happy, sad, surprise, etc., which are critical for depression recognition.

- **EEG/**  
  This folder provides all the necessary code for extracting emotional features from EEG data. It includes the implementation of a **ResNet-based model** that processes the EEG signals to capture emotional states. The code covers EEG data preprocessing, feature extraction, and model training to classify the emotional states of the subjects.

- **Machine learning/**  
  This folder contains the code and data for training and evaluating the **143 machine learning models** used in our study. These models were employed to validate the effectiveness of incorporating emotional features in depression diagnosis. The folder includes scripts for model selection, training, and performance comparison, covering models like Random Forest, Elastic Net, and SVM.

- **1D-2D-CNN-GRU/**  
  This folder provides the code and data for the **1D-2D-CNN-GRU model**, which is used for fusing emotional features from both EEG and audio data. This multimodal approach helps improve the robustness and accuracy of depression recognition. The folder also includes images demonstrating the model's performance, showing how the model achieved 91% accuracy on the test set. The code covers data fusion, model architecture, training, and evaluation.

## How to Use


1. **Data Preparation**  
   Each folder contains its respective datasets. Ensure the data is placed in the correct directories as described in the respective notebooks or scripts.

2. **Running the Models**  
   - To run the **audio feature extraction** using the 3D CNN-BLSTM model, navigate to the `Voice/` folder and execute the Jupyter notebook.
   - For **EEG feature extraction**, navigate to the `EEG/` folder and follow the instructions in the provided scripts.
   - To evaluate the **143 machine learning models**, navigate to the `Machine learning/` folder and execute the relevant scripts.
   - To run the **1D-2D-CNN-GRU multimodal model**, navigate to the `1D-2D-CNN-GRU/` folder and run the provided scripts for data fusion and training.

## Results

The **1D-2D-CNN-GRU model** achieved a training accuracy of 100% and a test set accuracy of **91.66%**, significantly outperforming traditional machine learning approaches on complex multimodal data. More detailed results and performance comparisons are included in the `1D-2D-CNN-GRU/` folder.

## Citation

If you find this repository useful in your research, please consider citing our work.
