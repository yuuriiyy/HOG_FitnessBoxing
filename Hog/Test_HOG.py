# Importing the necessary modules:
from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from skimage.io import imread
import joblib
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from HogFeatureExtractor import hog_feature
from skimage import io

# define parameters of HOG feature extraction
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
threshold = .3

# define paths to test images:
test_pos_im_path = r"../leapGestRecog/01/07_ok"  # Path to positive test images
test_neg_im_path = r"../leapGestRecog/01/03_fist"  # Path to negative test images

# Load the saved model
# model = joblib.load('svm_ok.npy')
model = joblib.load('model_name.npy')

# Prepare test data
test_data = []
test_labels = []

# Process positive test images
for file in os.listdir(test_pos_im_path):
    # img = Image.open(os.path.join(test_pos_im_path, file))
    # gray = img.convert('L')
    # fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)
    img = io.imread(os.path.join(test_pos_im_path, file))
    extractor = hog_feature(img)
    fd = extractor.extract_features()
    test_data.append(fd)
    test_labels.append(1)

# Process negative test images
for file in os.listdir(test_neg_im_path):
    # img = Image.open(os.path.join(test_neg_im_path, file))
    # gray = img.convert('L')
    # fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)
    img = io.imread(os.path.join(test_neg_im_path, file))
    extractor = hog_feature(img)
    fd = extractor.extract_features()
    test_data.append(fd)
    test_labels.append(0)

# Make predictions
predictions = model.predict(test_data)

# Evaluate the classifier
print("Evaluating classifier on test data ...")
print(classification_report(test_labels, predictions))