package com.example.fitnessboxing;

import android.content.res.AssetManager;
import android.util.Log;

import libsvm.*;

import org.apache.commons.io.IOUtils;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class SVMTrainer {


    public static svm_model main(String[] args, AssetManager assetManager, String PoseType) {

        if (!OpenCVLoader.initDebug()) {
            Log.e("SVMTrainer", "OpenCV initialization failed.");
        } else {
            Log.d("SVMTrainer", "OpenCV initialization succeeded.");
        }

        String posDirPath =  PoseType+ "pos"; // Directory containing positive images
        String negDirPath =  PoseType+ "neg"; // Directory containing negative images

        // Extract HoG features from positive images
        List<double[]> posFeatures = extractFeaturesFromDirectory(assetManager, posDirPath);
        double[] posLabels = new double[posFeatures.size()];
        Log.d("SVMTrainer", "Complete HoG Feature Extraction");
        // Set positive labels as 1
        for (int i = 0; i < posLabels.length; i++) {
            posLabels[i] = 1;
        }

        // Extract HoG features from negative images
        List<double[]> negFeatures = extractFeaturesFromDirectory(assetManager, negDirPath);
        double[] negLabels = new double[negFeatures.size()];
        // Set negative labels as -1
        for (int i = 0; i < negLabels.length; i++) {
            negLabels[i] = -1;
        }

        // Combine positive and negative features
        Log.d("SVMTrainer", "Combine positive and negative features");
        List<double[]> allFeatures = new ArrayList<>(posFeatures);
        allFeatures.addAll(negFeatures);
        double[] allLabels = new double[posFeatures.size() + negFeatures.size()];
        System.arraycopy(posLabels, 0, allLabels, 0, posLabels.length);
        System.arraycopy(negLabels, 0, allLabels, posLabels.length, negLabels.length);

        // Create a problem object
        Log.d("SVMTrainer", "Create a problem object");
        svm_problem problem = new svm_problem();
        problem.l = allLabels.length; // Number of training examples
        problem.y = allLabels; // Labels
        problem.x = new svm_node[allLabels.length][]; // Features

        // Fill the feature values
        Log.d("SVMTrainer", "Fill the feature values");
        for (int i = 0; i < allLabels.length; i++) {
            double[] features = allFeatures.get(i);
            problem.x[i] = new svm_node[features.length];
            for (int j = 0; j < features.length; j++) {
                svm_node node = new svm_node();
                node.index = j + 1; // Index should start from 1
                node.value = features[j];
                problem.x[i][j] = node;
            }
        }

        // Set SVM parameters (you may need to adjust these parameters based on your data)
        svm_parameter param = new svm_parameter();
        param.svm_type = svm_parameter.C_SVC;
        param.kernel_type = svm_parameter.RBF;
        param.C = 1;
        param.gamma = 0.5;

        // Train the SVM model
        svm_model model = svm.svm_train(problem, param);

        Log.d("SVMTrainer", "Training completed");
        return model;

//        // Save the trained model to a file (optional)
//
//        // Assuming you have the SVM model data in a byte array called svmModelData
//        String modelFileName = "svm_model.model";
//        ModelSaver.saveModelToDownloads(context, modelFileName, model);
//
//        // Display the path where the model is saved
//        File downloadsFolder = new File(context.getExternalFilesDir(null), "Download");
//        File modelFile = new File(downloadsFolder, modelFileName);
//        Log.d("SVMTrainer", "Model saved to: " + modelFile.getAbsolutePath());
    }



    // Function to extract HoG features from all images in a directory
    public static List<double[]> extractFeaturesFromDirectory(AssetManager assetManager, String dirPath) {
        List<double[]> features = new ArrayList<>();
        try {
            String[] files = assetManager.list(dirPath);
            Log.d("SVMTrainer", "File Path: " + dirPath);
            if (files != null) {
                for (String fileName : files) {
                    // Load the image as Mat
                    Log.d("SVMTrainer", "File name: " + fileName);
                    String filePath = dirPath + File.separator + fileName;
//                    InputStream inputStream = assetManager.open(filePath);
//                    Mat image = Imgcodecs.imdecode(new MatOfByte(IOUtils.toByteArray(inputStream)), Imgcodecs.IMREAD_COLOR);

                    // Open the input stream
                    InputStream inputStream = null;
                    try {
                        inputStream = assetManager.open(filePath);
                        Log.d("SVMTrainer", "Open file Successfully ");



                    // Check if input stream is not null
                    Mat image = null;
                    if (inputStream != null) {
                        Log.d("SVMTrainer", "inputStream != null");
                        try {
                            // Read bytes from input stream
                            byte[] bytes = IOUtils.toByteArray(inputStream);

                            // Check if bytes are read successfully
                            if (bytes != null) {
                                MatOfByte matOfByte = new MatOfByte(bytes);
                                image = Imgcodecs.imdecode(matOfByte, Imgcodecs.IMREAD_COLOR);
                                Log.d("SVMTrainer", "bytes != null");
                                if (!image.empty()) {
                                    Log.d("SVMTrainer", "Image converted successfully");
                                    HoGFeature hoGFeatureExtractor = new HoGFeature(image);
                                    double[] hogFeatures = hoGFeatureExtractor.extractFeatures();
                                    features.add(hogFeatures);
                                    if(hogFeatures.length > 0) {
                                        Log.d("SVMTrainer", "hogFeatures.length: " + hogFeatures.length);
                                        Log.d("SVMTrainer", "hogFeatures[0]" + hogFeatures[0]);
                                    }
                                    else{
                                        Log.e("SVMTrainer", "hogFeatures is null");
                                    }

                                }
                                else {
                                    Log.e("SVMTrainer", "Failed to decode image or image is empty");
                                }
                            } else {
                                Log.e("SVMTrainer", "Failed to read bytes from input stream");
                            }
                        } catch (IOException e) {
                            Log.e("SVMTrainer", "Error decoding image: " + e.getMessage());
                        } finally {
                            // Close the input stream
                            try {
                                inputStream.close();
                            } catch (IOException e) {
                                Log.e("SVMTrainer", "Error closing input stream: " + e.getMessage());
                            }
                        }
                    } else {
                        // Input stream is null, unable to open file
                        Log.e("SVMTrainer", "Input stream is null, unable to open file");
                    }
                    } catch (IOException e) {
                        Log.e("SVMTrainer", "Error opening file: " + e.getMessage());
                    }
                    
                }
            } else {
                Log.d("SVMTrainer", "No files found in " + dirPath);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return features;
    }

    public static void conversionCheck(InputStream inputStream){
        if (inputStream != null) {
            try {
                // Read bytes from input stream
                byte[] bytes = IOUtils.toByteArray(inputStream);

                // Check if bytes are read successfully
                if (bytes != null) {
                    // Decode the byte array into a Mat object
                    MatOfByte matOfByte = new MatOfByte(bytes);
                    Mat image = Imgcodecs.imdecode(matOfByte, Imgcodecs.IMREAD_COLOR);
                    if (image != null && !image.empty()) {
                        Log.d("SVMTrainer", "Image converted successfully");
                    } else {
                        Log.e("SVMTrainer", "Failed to decode image or image is empty");
                    }
                } else {
                    Log.e("SVMTrainer", "Failed to read bytes from input stream");
                }
            } catch (IOException e) {
                Log.e("SVMTrainer", "Error decoding image: " + e.getMessage());
            }
        } else {
            // Input stream is null, unable to open file
            Log.e("SVMTrainer", "Input stream is null, unable to open file");
        }
    }

}
