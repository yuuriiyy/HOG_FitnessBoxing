from skimage import color
from skimage.feature import hog
from HogFeatureExtractor import hog_feature
import cv2
import joblib
import numpy as np


# Load the saved model and video
model = joblib.load('./model_ped.npy')
vid = cv2.VideoCapture("./ped_video2.mov")

prev_roi = None
motion_detected = False

while vid.isOpened():
    ret, frame = vid.read()
    if not ret:
        break  # If no frame is read, break out of the loop
    
    predicted_label = "Negative"
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    start_point = (500, 300)
    end_point = (700, 700)
    roi = gray[start_point[1]:end_point[1], start_point[0]:end_point[0]]
    cv2.imshow("roi", roi)
    
    if prev_roi is not None:
        # Calculate the absolute difference between the current ROI and the previous ROI
        diff = cv2.absdiff(prev_roi, roi)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        # print(np.sum(thresh))
        motion_detected = np.sum(thresh) > 720000

    prev_roi = roi.copy()

    if motion_detected:
        # Resize the ROI to match the input size expected by your HOG feature extractor
        roi_resized = cv2.resize(roi, (64, 128))  # Assuming the expected size is (64, 128)
        # fd = hog(roi_resized, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2', feature_vector=True)
        extractor = hog_feature(roi_resized)
        fd = extractor.extract_features()
        prediction = model.predict([fd])
        predicted_label = "Positive" if prediction[0] == 1 else "Negative"
        cv2.putText(frame, predicted_label, (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.rectangle(frame, start_point, end_point, (0, 0, 255), 2)
    cv2.imshow("Frame", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
vid.release()
cv2.destroyAllWindows()