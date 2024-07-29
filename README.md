# Breast Cancer Detection Project

## Overview
This project aims to develop a web application capable of segmenting and classifying tumors in ultrasound data, enabling the identification of malignant or benign tumors in patients. The application uses machine learning models for classification and a U-Net model for segmentation.

## Team
- Yeşim Tortop
- Mustafa Soydan
- Kerem Erciyes

## Supervisor
- Prof. Dr. Isabella Castiglioni

## Project Structure
The project includes several components:
- **User Interfaces**: Implemented using Streamlit for easy navigation and interaction.
- **Classification Models**: SVM, Random Forest (RF), and Multi-Layer Perceptron (MLP). The MLP model was selected as the best model.
- **Segmentation Model**: U-Net model for image segmentation.

## User Interface Files
1. **main.py**:
    - The main entry point of the application.
    - Manages navigation between different pages (Login, Home, Radiology).

2. **login.py**:
    - Handles the login functionality.
    - Simple hardcoded login credentials for demonstration purposes.

3. **home.py**:
    - Home page after successful login.
    - Displays patient data and allows navigation to the Radiology page.

4. **radiology.py**:
    - Main page for radiology functionalities.
    - Allows uploading radiological images, performs segmentation using U-Net, extracts features using Pyradiomics, and classifies the image as benign or malignant using the MLP model.

## Models and Scripts
1. **feature_extraction_RF_MLP(NN).ipynb**:
    - Notebook for feature extraction and training the RF and MLP models.

2. **UNet_segmantation.ipynb**:
    - Notebook for training and saving the U-Net segmentation model.

3. **SVM_balanced_data.ipynb**:
    - Notebook for training the SVM model with balanced data.

## Model Selection
- **Classification**: After experimenting with SVM, RF, and MLP, the MLP model was selected for its superior performance.
- **Segmentation**: The U-Net model was chosen for its effectiveness in biomedical image segmentation.

## Usage
1. **Login**: Use the login interface to access the application.
2. **Home Page**: View and manage patient data. Navigate to the Radiology page.
3. **Radiology Page**: Upload a radiological image. The image is processed through the following steps:
    - **Segmentation**: The U-Net model segments the image.
    - **Feature Extraction**: Extract features using Pyradiomics.
    - **Classification**: The MLP model classifies the tumor as benign or malignant.

## Ethical Considerations
- Adheres to GDPR and national guidelines.
- Ensures data minimization, anonymization, and data security.
- Informed consent and transparent outputs are prioritized.

## Conclusion
The project provides a reliable and ethically sound method for diagnosing breast health from ultrasound data. The use of AI in tumor detection and classification aids in early diagnosis and monitoring.

For more detailed information, refer to the provided PDF presentation: `presentation_of_breast_cancer_classification_segmentation.pdf`.

---

This README provides an overview of the project, its components, and instructions on how to use the application. For any further queries or detailed explanations, please refer to the project documentation and notebooks.
