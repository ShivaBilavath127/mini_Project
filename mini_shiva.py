import os 
import cv2 
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score, classification_report 
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt 
 
dataset_dir = "C:/Users/SHIVA/Downloads/eye disease dataset/dataset" 
IMAGE_SIZE = (128, 128)  # Resize images to 128x128 pixels 
 
def load_images_from_directory(directory, img_size=IMAGE_SIZE): 
    images = [] 
    labels = [] 
    class_names = os.listdir(directory) 
 
    for class_name in class_names: 
        class_dir = os.path.join(directory, class_name) 
        if os.path.isdir(class_dir): 
            for img_name in os.listdir(class_dir): 
                img_path = os.path.join(class_dir, img_name) 
                img = cv2.imread(img_path) 
                if img is not None: 
                    img = cv2.resize(img, img_size) 
                    img = img / 255.0  # Normalize pixel values  
                    images.append(img) 
                    labels.append(class_name) 
     
    return np.array(images), np.array(labels) 
 
# Load training images and labels 
#train_images, train_labels = load_images_from_directory(dataset_dir) 
# Load a smaller subset for quick testing 
train_images, train_labels = load_images_from_directory(dataset_dir)[:1000] 
 
# Check shapes 
print("Train Images Shape:", train_images.shape) 
print("Train Labels Shape:", train_labels.shape) 
 
n_samples = len(train_images) 
flattened_images = train_images.reshape(n_samples, -1) 
 
# Label encoding 
le = LabelEncoder() 
train_labels_encoded = le.fit_transform(train_labels) 
 
# Split data into training and validation sets 
X_train, X_val, y_train, y_val = train_test_split(flattened_images, 
train_labels_encoded, test_size=0.2, random_state=42) 
 
print("Training Data Shape:", X_train.shape) 
print("Validation Data Shape:", X_val.shape) 
 
scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train) 
X_val = scaler.transform(X_val) 
 
# Optional: Use PCA for dimensionality reduction 
pca = PCA(n_components=50) 
X_train_pca = pca.fit_transform(X_train) 
X_val_pca = pca.transform(X_val) 
print("PCA-Reduced Training Data Shape:", X_train_pca.shape) 
# Train a Support Vector Machine (SVM) classifier (or) Random Forest Classifier 
#svm_classifier = SVC(kernel='linear', probability=True) 
#svm_classifier.fit(X_train_pca, y_train) 
# Train a Random Forest classifier 
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42) 
rf_classifier.fit(X_train_pca, y_train) 
# Make predictions on the validation set 
y_pred = rf_classifier.predict(X_val_pca) 
# Print accuracy and classification report 
accuracy = accuracy_score(y_val, y_pred) 
print(f"Validation Accuracy: {accuracy}") 
print("Classification Report:") 
print(classification_report(y_val, y_pred, target_names=le.classes_)) 
# Path to new image for prediction (replace with your own path) 
img_path = "C:/Users/SHIVA/Downloads/eye disease dataset/online_img.jpg" 
# Preprocess the image 
img = cv2.imread(img_path) 
img = cv2.resize(img, IMAGE_SIZE) 
img = img / 255.0  # Normalize 
img_flattened = img.reshape(1, -1) 
img_scaled = scaler.transform(img_flattened) 
img_pca = pca.transform(img_scaled) 
# Predict the class of the new image 
predicted_label_index = rf_classifier.predict(img_pca) 
predicted_label = le.inverse_transform(predicted_label_index) 
print("Predicted class:", predicted_label[0]) 