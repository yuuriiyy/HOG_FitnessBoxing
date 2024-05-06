import cv2

image = cv2.imread('car_image.png')
plate_image = cv2.imread('plate_image.png')

gray_plate_image = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)

cv2.imshow('gray_plate_image', gray_plate_image)

sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray_plate_image, None)
image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)

cv2.imshow('Keypoints', image_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
