import matplotlib.pyplot as plt
from skimage import io
import imageio
from skimage import color
from skimage.transform import resize
import math
from skimage.feature import hog
import numpy as np

class HogFeatureExtractor:
    def __init__(self, img):
        self.img = img

    def _calculate_mag_theta(self, img):
        mag = []
        theta = []
        for i in range(128):
            magnitudeArray = []
            angleArray = []
            for j in range(64):
                # Condition for axis 0
                if j-1 <= 0 or j+1 >= 64:
                    if j-1 <= 0:
                        # Condition if first element
                        Gx = img[i][j+1] - 0
                    elif j + 1 >= len(img[0]):
                        Gx = 0 - img[i][j-1]
                # Condition for first element
                else:
                    Gx = img[i][j+1] - img[i][j-1]
                
                # Condition for axis 1
                if i-1 <= 0 or i+1 >= 128:
                    if i-1 <= 0:
                        Gy = 0 - img[i+1][j]
                    elif i +1 >= 128:
                        Gy = img[i-1][j] - 0
                else:
                    Gy = img[i-1][j] - img[i+1][j]

                # Calculating magnitude
                magnitude = math.sqrt(pow(Gx, 2) + pow(Gy, 2))
                magnitudeArray.append(round(magnitude, 9))

                # Calculating angle
                if Gx == 0:
                    angle = math.degrees(0.0)
                else:
                    angle = math.degrees(abs(math.atan(Gy / Gx)))
                angleArray.append(round(angle, 9))
            mag.append(magnitudeArray)
            theta.append(angleArray)

        mag = np.array(mag)
        theta = np.array(theta)
        return mag, theta


    def extract_features(self):
        img = self.img
        number_of_bins = 9
        step_size = 180 / number_of_bins
        mag, theta = self._calculate_mag_theta(img)

        return feature_vectors


# Example of using the class
if __name__ == "__main__":
    image_path = "./p1.png"  # Path to your single image
    image = io.imread(image_path)[:,:,:3]
    image = color.rgb2gray(image)
    image = resize(image, (128, 64))
    extractor = HogFeatureExtractor(image)
    features = extractor.extract_features()
    print("Feature vectors extracted successfully.")
    print(features)
