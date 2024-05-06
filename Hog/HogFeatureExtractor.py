import matplotlib.pyplot as plt
from skimage import io
from skimage import color
from skimage.transform import resize
import math
import numpy as np

class hog_feature:
    def __init__(self, img):
        if img.ndim == 3:
            self.img = color.rgb2gray(img).astype(np.float64)
        else:
            self.img = img.astype(np.float64)

    def _calculate_mag_theta(self, img):
        mag = np.zeros_like(img, dtype = float)
        theta = np.zeros_like(img, dtype = float)
        for i in range(1, img.shape[0] - 1):
            for j in range(1, img.shape[1] - 1):
                Gx = img[i, j + 1] - img[i, j - 1]
                Gy = img[i - 1, j] - img[i + 1, j]
                mag[i, j] = np.sqrt(Gx**2 + Gy**2)
                theta[i, j] = np.degrees(np.arctan2(Gy, Gx))

        
        return mag, theta

    def extract_features(self):
        img = self.img
        number_of_bins = 9
        step_size = 180 / number_of_bins
        cell_size = 8
        mag, theta = self._calculate_mag_theta(img)
        feature_vectors = []


        for i in range(0, img.shape[0] - cell_size, cell_size):
            for j in range(0, img.shape[1] - cell_size, cell_size):
                cell_magnitude = mag[i:i + cell_size, j:j + cell_size]
                cell_orientation = theta[i:i + cell_size, j:j + cell_size]
                histogram = np.zeros(number_of_bins)
                
                for m in range(cell_size):
                    for n in range(cell_size):
                        angle = cell_orientation[m, n]
                        # if(angle < 0): print("get")
                        magnitude = cell_magnitude[m, n]
                        bin_index = int(angle / step_size) % number_of_bins
                        # print("bin_index: ", bin_index)
                        histogram[bin_index] += magnitude
                
                feature_vectors.extend(histogram)


        # Normalize the feature vectors
        feature_vectors = np.array(feature_vectors)
        feature_vectors /= np.linalg.norm(feature_vectors) + 1e-5
        return feature_vectors