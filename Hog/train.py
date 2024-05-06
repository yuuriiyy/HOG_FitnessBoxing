# Importing the necessary modules:
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from skimage import color
from skimage import io
import joblib
import numpy as np
import os
from numpy import *
from HogFeatureExtractor import hog_feature

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
# from HogFeatureExtractor_g import hog_feature

# define parameters of HOG feature extraction
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
threshold = .3


# # define path to images:
# pos_im_path = r"../leapGestRecog/00/07_ok" # This is the path of our positive input dataset
# # define the same for negatives
# neg_im_path= r"../leapGestRecog/00/03_fist"

# pos_im_path = "../INRIAPerson/train_64x128_H96/pos"  # Hardcoded path for positive images
# neg_im_path = "../INRIAPerson/train_64x128_H96/neg"  # Hardcoded path for negative images

pos_im_path = "../images/left"
neg_im_path = "../images/nothing"
model_name = "model_left_punch.npy"

# read the image files:
pos_im_listing = os.listdir(pos_im_path) # it will read all the files in the positive image path (so all the required images)
neg_im_listing = os.listdir(neg_im_path)
num_pos_samples = size(pos_im_listing) # simply states the total no. of images
num_neg_samples = size(neg_im_listing)
print("postive samples: ", num_pos_samples) # prints the number value of the no.of samples in positive dataset
print("negative samples: ", num_neg_samples)
data= []
labels = []

# compute HOG features and label them:

for file in pos_im_listing: #this loop enables reading the files in the pos_im_listing variable one by one
    # img = Image.open(pos_im_path + '/' + file) # open the file
    img = io.imread(pos_im_path + '/' + file)
    img = resize(img, (64, 128))

    if img.ndim == 3:
        img = color.rgb2gray(img)
    # fd = hog(img, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)# fd= feature descriptor
    extractor = hog_feature(img)
    fd = extractor.extract_features()
    data.append(fd)
    labels.append(1)
    
# Same for the negative images
for file in neg_im_listing:
    # img= Image.open(neg_im_path + '/' + file)
    img = io.imread(neg_im_path + '/' + file)
    img = resize(img, (64, 128))
    if img.ndim == 3:
        img = color.rgb2gray(img)
    # fd = hog(img, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True) 

    extractor = hog_feature(img)
    fd = extractor.extract_features()
    data.append(fd)
    labels.append(0)
# encode the labels, converting them from strings to integers
le = LabelEncoder()
labels = le.fit_transform(labels)

#%%
# Partitioning the data into training and testing splits, using 80%
# of the data for training and the remaining 20% for testing
print(" Constructing training/testing split...")
(trainData, testData, trainLabels, testLabels) = train_test_split(
	np.array(data), labels, test_size=0.20, random_state=42)
#%% Train the linear SVM
print(" Training Linear SVM classifier...")
model = LinearSVC()
model.fit(trainData, trainLabels)
#%% Evaluate the classifier
print(" Evaluating classifier on test data ...")
predictions = model.predict(testData)
print(classification_report(testLabels, predictions))

# Generate confusion matrix
cm = confusion_matrix(testLabels, predictions)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')
plt.show()






#%% Save the Model
joblib.dump(model, model_name)