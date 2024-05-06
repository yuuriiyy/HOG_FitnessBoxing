import cv2
import pysift

image = cv2.imread('car_image.png', 0)
keypoints, descriptors = pysift.computeKeypointsAndDescriptors(image)

print(keypoints, descriptors)