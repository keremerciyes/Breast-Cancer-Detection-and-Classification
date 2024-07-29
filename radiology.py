#!/Library/Frameworks/Python.framework/Versions/3.11/bin/python3
from sklearn.discriminant_analysis import StandardScaler
import streamlit as st
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import radiomics
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
from radiomics import featureextractor
import SimpleITK as sitk
import sys
import pandas as pd
print(sys.executable)

# Load models
segmentation_model_path ='/Users/mustafasoydan/Desktop/projects/healthcare/best_segmentation_model.pth'
classification_model_path = '/Users/mustafasoydan/Desktop/projects/healthcare/best_classification_model.pth'


# Parameters
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 1
EPOCHS = 10
LEARNING_RATE = 0.1

# Custom dataset class
class CancerDataset(Dataset):
    def __init__(self, image_files=None, image_data=None, transform=None):
        self.image_files = image_files
        self.image_data = image_data
        self.transform = transform

        if self.image_files is not None:
            self.image_files = [f for f in self.image_files if f.endswith('.png') and not f.endswith('_mask.png')]

    def __len__(self):
        return len(self.image_files) if self.image_files is not None else len(self.image_data)

    def __getitem__(self, idx):
        if self.image_files is not None:
            img_name = self.image_files[idx]
            img_path = img_name
            image = Image.open(img_path).convert('L')  # Convert image to grayscale
        else:
            image = Image.open(self.image_data[idx]).convert('L')

        if self.transform:
            image = self.transform(image)

        return image

# Define transformations
data_transforms = {
    'test': transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3], std=[0.2])
    ])
}

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoder = nn.ModuleList([
            DoubleConv(in_channels, 64),
            DoubleConv(64, 128),
            DoubleConv(128, 256),
            DoubleConv(256, 512),
        ])

        self.bottleneck = DoubleConv(512, 1024)

        self.upconvs = nn.ModuleList([
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
        ])

        self.decoder = nn.ModuleList([
            DoubleConv(1024, 512),
            DoubleConv(512, 256),
            DoubleConv(256, 128),
            DoubleConv(128, 64),
        ])

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc_feats = []
        for enc in self.encoder:
            x = enc(x)
            enc_feats.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.bottleneck(x)

        for i in range(len(self.upconvs)):
            x = self.upconvs[i](x)
            enc_feat = enc_feats[-(i + 1)]
            x = torch.cat([x, enc_feat], dim=1)
            x = self.decoder[i](x)

        x = self.final_conv(x)
        return x

# Classification model
class ComplexMLPModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate):
        super(ComplexMLPModel, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)
        prev_size = input_size

        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size

        self.output_layer = nn.Linear(prev_size, output_size)

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
            x = self.dropout(x)
        x = self.output_layer(x)
        return x


def convert_to_grayscale(image):
    """
    Convert a color image (RGB or RGBA) to grayscale if necessary.
    """
    if image.GetNumberOfComponentsPerPixel() > 1:
        return sitk.VectorMagnitude(image)
    return image


def radiology_page():
    st.title("Radiology Page")
    uploaded_file = st.file_uploader("Upload Radiological Image", type=["jpg", "png", "jpeg", "dcm"])

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("Interrupt", key="button6"):
            st.session_state['interrupted'] = True

    with col2:
        if st.button('Run', key="button7"):
            st.session_state['interrupted'] = False

    if 'interrupted' not in st.session_state:
        st.session_state['interrupted'] = False

    if uploaded_file is not None and not st.session_state['interrupted']:
        size= (128,128)
        image = Image.open(uploaded_file).convert('L') # convert grayscale
        #image = convert_to_grayscale(image)
        image = image.resize(size)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        original_image = image.save("original.png")
        
        st.write("Performing Segmentation...")
        
        transform = transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.3], std=[0.2])
        ])
        image_tensor = transform(image).unsqueeze(0)
        print('transfomraiton done') 
        segmentation_model = UNet(in_channels=1, out_channels=1)
        segmentation_model.load_state_dict(torch.load(segmentation_model_path, map_location=torch.device('cpu')))
        segmentation_model.eval()
        with torch.no_grad():
            mask = segmentation_model(image_tensor)
            mask = torch.sigmoid(mask)
        print('segmentatipn done')
        
        mask_image = transforms.ToPILImage()(mask.squeeze(0))
        #mask_image = mask
        #mask_image = convert_to_grayscale(mask_image)
        mask_image.save("segmented_mask.png")
        st.image(mask_image, caption="Segmented Mask", use_column_width=True)
        print('masking done')
        # Feature Extraction
        st.write("Extracting Features using Pyradiomics...")

        params = {
            #'label': 255,
            'enableCExtensions': True,
            'additionalInfo': True,
            'force2D': True,
            'force2Ddimension': 2
        }
        extractor = featureextractor.RadiomicsFeatureExtractor(**params)



        print('featureextractor worked')

        original_image_path = '/Users/mustafasoydan/Desktop/projects/healthcare/original.png'
        mask_image_path = '/Users/mustafasoydan/Desktop/projects/healthcare/segmented_mask.png'

        image_array = sitk.ReadImage(original_image_path)
        mask_array = sitk.ReadImage(mask_image_path)

        image_array = convert_to_grayscale(image_array)
        mask_array = convert_to_grayscale(mask_array)


        image_array2 = sitk.GetArrayFromImage(image_array)
        mask_array2 = sitk.GetArrayFromImage(mask_array)
        print('array dimension control')
        print(image_array2.size)
        print(mask_array2.size)

        features = extractor.execute(image_array, mask_array)
        print('featurelar cikti')

        #print(features)

        # Normalize features into a consistent format
        feature_dict = {key: [value] if not isinstance(value, np.ndarray) else value.flatten().tolist() for key, value in features.items()}
        

        df = pd.DataFrame(feature_dict)
        print('featurelar dataframe e çevirildi')
        print(df)

        df.to_csv('deneme.csv', index = False)  

        data = pd.read_csv('/Users/mustafasoydan/Desktop/projects/healthcare/deneme.csv') 


        columns_to_drop = ['diagnostics_Image-original_Dimensionality', 'diagnostics_Image-original_Spacing',
                   'diagnostics_Versions_PyRadiomics', 'diagnostics_Versions_Numpy',
                   'diagnostics_Versions_SimpleITK', 'diagnostics_Versions_PyWavelet',
                   'diagnostics_Versions_Python', 'diagnostics_Configuration_Settings',
                   'diagnostics_Configuration_EnabledImageTypes', 'diagnostics_Image-original_Minimum','diagnostics_Image-original_Maximum']

        # Drop the specified columns from X
        X = data.drop(columns=columns_to_drop)
        # Iterate over columns in X
        for column in X.columns:
        # Check if any value in the column is of string type
            if X[column].apply(lambda x: isinstance(x, str)).any():
        # Drop the column if it contains strings
                X.drop(column, axis=1, inplace=True)

        
        mean = pd.read_csv('/Users/mustafasoydan/Desktop/projects/healthcare/mean_new.csv', header=None)
        print('mean shape', mean.shape)
        #mean = mean.squeeze() # 1D array
        std = pd.read_csv('/Users/mustafasoydan/Desktop/projects/healthcare/std_new.csv', header=None)
        print('std shape', std.shape)
        #std = std.squeeze() # 1D array
        

        #scaler = StandardScaler()
        #X_normalized_cleaned = scaler.fit_transform(X)
        #X_normalized_cleaned = (X-X.mean()) / X.std()
        
        #X_normalized_cleaned = X
        #print(X_normalized_cleaned.shape)
        #print(X_normalized_cleaned)
        X_normalized_cleaned = X.to_numpy()
        #print('X normalized shape : ', X_normalized_cleaned.shape)

        #X_normalized_cleaned = (X_normalized_cleaned - mean) / std
        #print('SON',X_normalized_cleaned)
        #X_normalized_cleaned = X_normalized_cleaned.to_numpy()

        #feature_vector = [features[key] for key in features.keys() if isinstance(features[key], (int, float))]
        #feature_tensor = torch.tensor(feature_vector).unsqueeze(0)
        #print('feature vector created')

        # Classification
        st.write("Classifying...")
        hidden_size = 486  # Adjust the number of hidden units
        hidden_layers = 4
        hidden_sizes = [hidden_size] * hidden_layers
        dropout_rate = 0.24569627766
        # Initialize the model

        input_size = 96
        output_size = 2  # Number of classes

        classification_model = ComplexMLPModel(input_size, hidden_sizes, output_size, dropout_rate)
        print('model defined correctly')
        classification_model.load_state_dict(torch.load(classification_model_path))#, map_location=torch.device('cpu')))
        print('model fetched correctly')
        classification_model.eval()
        print('model evaluated correctly')
        #classification_model.eval()
        print(X_normalized_cleaned)


        X_tensor = torch.tensor(X_normalized_cleaned, dtype=torch.float32)

        #X_tensor = (X_tensor - mean) / std

        print('X tensor: \n')
        print(X_tensor)
        print(X_tensor.shape)
        with torch.no_grad():
            output = classification_model(X_tensor)
            print('output: ', output)
            _, predicted = torch.max(output, 1)
            print(predicted)
            predicted_class = 'malignant' if predicted.item() == 1 else 'benign'
            print(f'The predicted class for the input data is: {predicted_class}')

                # accuracy = accuracy_score(y_test_tensor, predicted)
            #prediction = torch.argmax(output, dim=1).item()
        #print('result is:', result)
        #result = "Malignant" if predicted == 1 else "Benign"
        st.success(f"The uploaded radiological image is classified as: {predicted_class}")
    
if 'logged_in' in st.session_state and 'selected_patient' in st.session_state:
    if st.session_state["logged_in"] and st.session_state["selected_patient"]:
        radiology_page()
    else:
        st.error("Please login and select a patient first.")
else:
    st.error("Please login and select a patient first.")