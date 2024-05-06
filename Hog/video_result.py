from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from skimage.io import imread
import joblib
import numpy as np
from PIL import Image
from HogFeatureExtractor import hog_feature
from skimage import io, color
import cv2


# Define parameters of HOG feature extraction
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
threshold = .3

#%%
# Load the saved model and preprocess the single test image
vid = cv2.VideoCapture("./ped_video.mov") 
model = joblib.load('./model_pedestrian.npy')
image_path = "./ped.png"  # Change this to the path of your image
img = io.imread(image_path)
img = np.resize(img, (128, 64))
if img.ndim == 3:
    if img.shape[2] == 4:  # Check if the image is RGBA
        img = img[:, :, :3]  # Drop the alpha channel
    img = color.rgb2gray(img)  # Convert the image to grayscale

#%%
# Extract HoG Features
extractor = hog_feature(img)
fd = extractor.extract_features()
# fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)
test_data = [fd]
# Make prediction
prediction = model.predict(test_data)
predicted_label = "Positive" if prediction[0] == 1 else "Negative"




while vid.isOpened():
    ret, frame = vid.read() 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    start_point = (500, 300) 
    end_point = (700, 700) 
    color = (255, 0, 0) 
    thickness = 2
    image = cv2.rectangle(frame, start_point, end_point, color, thickness)
    cv2.imshow("output", image)

    # fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)
    # test_data = [fd]
    # prediction = model.predict(test_data)
    # predicted_label = "Positive" if prediction[0] == 1 else "Negative"
    # print(predicted_label)
    # frame =  cv2.putText(frame, predicted_label, (30, 30), cv2.FONT_HERSHEY_SIMPLEX , 1, 
    #               (255, 255, 0) , 2, cv2.LINE_AA, True)
    # cv2.imshow('frame', frame) 

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

# Decode the predicted label (if necessary)
vid.release() 
cv2.destroyAllWindows() 