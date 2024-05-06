package com.example.fitnessboxing;

import static java.lang.Math.abs;

import android.util.Log;

import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Mat;
import org.opencv.core.CvType;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

public class HoGFeature {
    private Mat img;
    public Mat resizedImg = new Mat();

    public HoGFeature(Mat img) {
        if (img == null || img.empty()) {
            Log.e("HoGFeature", "Input image is null or empty");
            return;
        }
        this.img = img.clone(); // Make a deep copy to avoid modifying the original image
        Size newSize = new Size(64, 128);
        Imgproc.resize(this.img, resizedImg, newSize);
//        this.img.convertTo(this.img, CvType.CV_64FC1); // Convert to float64
    }

    private double[][] calculateMagnitudeAndTheta(Mat img) {
        double[] mag = new double[img.rows() * img.cols()];
        double[] theta = new double[img.rows() * img.cols()];

        for (int i = 1; i < img.rows() - 1; i++) {
            for (int j = 1; j < img.cols() - 1; j++) {
                double Gx = img.get(i, j + 1)[0] - img.get(i, j - 1)[0];
                double Gy = img.get(i - 1, j)[0] - img.get(i + 1, j)[0];
                int index = i * img.cols() + j;
                mag[index] = Math.sqrt(Math.pow(Gx, 2) + Math.pow(Gy, 2));
                theta[index] = Math.toDegrees(Math.atan2(Gy, Gx));
            }
        }

        return new double[][] { mag, theta };
    }

    public double[] extractFeatures() {
        int numberOfBins = 9;
        double stepSize = 180.0 / numberOfBins;
        int cellSize = 8;
        double[][] magTheta = calculateMagnitudeAndTheta(resizedImg);
        double[] mag = magTheta[0];
        double[] theta = magTheta[1];
        List<Double> featureVectors = new ArrayList<>();

        for (int i = 0; i < resizedImg.rows() - cellSize; i += cellSize) {
            for (int j = 0; j < resizedImg.cols() - cellSize; j += cellSize) {
                double[] histogram = new double[numberOfBins];


                for (int m = 0; m < cellSize; m++) {
                    for (int n = 0; n < cellSize; n++) {
                        int index = (i + m) * resizedImg.cols() + (j + n);
                        double angle = theta[index];
                        double magnitude = mag[index];
                        int binIndex = (int) (abs(angle) / stepSize) % numberOfBins;
                        histogram[binIndex] += magnitude;

                    }
                }

                for (double value : histogram) {
                    featureVectors.add(value);
                }
            }
        }

        // Normalize the feature vectors
        double[] featureArray = new double[featureVectors.size()];
        for (int i = 0; i < featureVectors.size(); i++) {
            featureArray[i] = featureVectors.get(i);
        }

        double norm = 0;
        for (double value : featureArray) {
            norm += Math.pow(value, 2);
        }
        norm = Math.sqrt(norm);

        for (int i = 0; i < featureArray.length; i++) {
            featureArray[i] /= (norm + 1e-5);
        }

        return featureArray;
    }
}
