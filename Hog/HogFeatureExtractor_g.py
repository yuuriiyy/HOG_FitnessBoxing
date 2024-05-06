import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import copy
import cv2
import math


class hog_feature:
    def __init__(self, img):
        self.img = img.astype(np.float64) 


    def gradient(self):
        # image2 = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        image2 = self.img
        mag = np.zeros(np.shape(image2))
        angle = np.zeros(np.shape(image2))
        bigs = np.zeros(np.shape(image2))
        for i in range(len(image2)):
            for j in range(len(image2[0])):
                if(i-1>= 0 and i+1<len(image2)):
                    gx = image2[i+1][j]-image2[i-1][j]
                elif(i-1<=0):
                    gx = image2[i+1][j] - 0
                elif(i+1>=len(image2)):
                    gx = image2[i-1][j] - 0
                    
                if(j-1>= 0 and j+1<len(image2[0])):
                    gy = image2[i][j+1]-image2[i][j-1]
                elif(j-1<=0):
                    gy = image2[i][j+1] - 0
                elif(j+1>=len(image2[0])):
                    gy = image2[i][j-1] - 0
                mag[i][j] = math.sqrt(gx**2 + gy**2)
                bigs[i][j] = image2[i][j]
                temp = 0
                if(gx == 0):
                    temp = 90
                else:
                    temp = abs(math.atan(gy/gx))
                
                angle[i][j] = int(math.degrees(temp)) #radians

        return mag, angle, bigs

                
    def resize(self, image):
        reimg = cv2.resize(image, (128,64))
        return reimg

    def extract_features(self):
        mag, angle, bigs = self.gradient()
        nxblock = int(len(mag)/8)
        nyblock = int(len(mag[0])/8)
        histograms  = np.zeros((nxblock*nyblock, 9))
        fv = np.zeros(((nxblock-1),(nyblock-1), 36))
        
        for i in range(nxblock):
            for j in range(nyblock):
                for k in range(8):
                    for l in range(8):
                        anglebin= int(angle[k*i][l*j]/20)%9
                        weight = (angle[k*i][l*j]%20)/20
                        histograms[i*j][anglebin] += weight * mag[k*i][l*j]
                        histograms[i*j][(anglebin+1)%9] += (1-weight) * mag[k*i][l*j]
        for i in range(nxblock-1):
            for j in range(nyblock-1):
                z = 0
                norm = 0
                temp = np.zeros(36)
                for k in range(2):
                    for l in range(2):
                        for n in range(9):
                            temp[z] =  histograms[(i+k)*(j+l)][n]
                            norm += temp[z]**2
                            z += 1
                math.sqrt(norm)
                temp = temp/norm
                fv[i][j] = temp
                        
                        
        return fv