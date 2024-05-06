package com.example.fitnessboxing;

import android.content.res.AssetManager;
import android.util.Log;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;

import org.apache.commons.io.IOUtils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.imgcodecs.Imgcodecs;


public class FolderTest {

    public static void main(String[] args, AssetManager assetManager) {
        // Directory paths
        String posDirPath = "pos"; // Directory containing positive images
        String negDirPath = "neg"; // Directory containing negative images

        testFolder(assetManager, posDirPath, "pos");
        testFolder(assetManager, negDirPath, "neg");

    }

    // Function to test if a folder exists and print the total number of pictures
    private static void testFolder(AssetManager assetManager, String dirPath, String folderName) {
        Mat image = null;
        try {
            String[] files = assetManager.list(dirPath);
            if (files != null) {
                int numPictures = files.length;
                Log.d("testFolder", "Folder '" + folderName + "' exists. Total number of pictures: " + numPictures);
            } else {
                Log.d("testFolder", "Folder '" + folderName + "' is empty.");
            }
        } catch (IOException e) {
            Log.d("testFolder", "Folder '" + folderName + "' does not exist.");
        }
    }

}
