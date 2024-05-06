from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from skimage.io import imread
import joblib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from HogFeatureExtractor import hog_feature
from skimage import io, color


# Define parameters of HOG feature extraction
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
threshold = .3
window_size = (64, 64)  # Define window size
stride = (8, 8)  # Define stride

#%%
# Load the saved model and preprocess the single test image
model = joblib.load('./model_ped.npy')
image_path = "./not_ped.png"  # Change this to the path of your image
# image_path = "./ped.png"  # Change this to the path of your image
img = io.imread(image_path)
img = np.resize(img, (64, 128))
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



#%%
# Open the image using PIL
image = Image.open(image_path)
draw = ImageDraw.Draw(image)
text = f"Predicted Label: {predicted_label}"
position = (10, 10)  # Adjust the position as needed
font_size = 40  # Increase this value for larger text
font = ImageFont.load_default(font_size)  # Using the default font provided by PIL
draw.text(position, text, fill="red", font=font)
image.show()
print(predicted_label)