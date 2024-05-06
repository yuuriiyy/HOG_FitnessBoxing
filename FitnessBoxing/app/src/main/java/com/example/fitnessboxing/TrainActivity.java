package com.example.fitnessboxing;

import libsvm.*;

import static org.opencv.android.Utils.matToBitmap;

import android.content.Context;
import android.content.Intent;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.media.MediaPlayer;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.util.Log;


import androidx.appcompat.app.AppCompatActivity;

import org.apache.commons.io.IOUtils;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;

public class TrainActivity extends AppCompatActivity {



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_train);
        Button startButton = findViewById(R.id.start_button);
        Button homeButton = findViewById(R.id.home_button);
        AssetManager assetManager = this.getAssets();
        Log.d("TrainActivity_DBG", "assetManager: " + assetManager);

        homeButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // Create an Intent to navigate back to MainActivity
                Intent intent = new Intent(TrainActivity.this, MainActivity.class);
                startActivity(intent); // Start the MainActivity
            }
        });


        startButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Context context = v.getContext(); // Get the Context from the clicked View
                AssetManager assetManager = context.getAssets();

                if (assetManager != null) {
                    FolderTest.main(null, assetManager);
                    Log.d("TrainActivity_DBG", "Start Training");
                    svm_model model = SVMTrainer.main(null, assetManager, "right");

                    try {
                        String dirPath = "test";
                        String[] files = assetManager.list(dirPath);
                        if (files != null && files.length > 0) {
                            for (String fileName : files) {
                                Log.d("TrainActivity_DBG", "FileName: " +  fileName);
                                String filePath = dirPath + File.separator + fileName;
                                InputStream inputStream = null;
                                inputStream = assetManager.open(filePath);
                                Mat image = Imgcodecs.imdecode(new MatOfByte(IOUtils.toByteArray(inputStream)), Imgcodecs.IMREAD_COLOR);
                                HoGFeature hoGFeatureExtractor = new HoGFeature(image);
                                double[] hogFeatures = hoGFeatureExtractor.extractFeatures();
                                // Convert features to svm_node format
                                svm_node[] nodes = new svm_node[hogFeatures.length];
                                for (int i = 0; i < hogFeatures.length; i++) {
                                    svm_node node = new svm_node();
                                    node.index = i + 1; // Index starts from 1
                                    node.value = hogFeatures[i];
                                    nodes[i] = node;
                                }

                                // Predict using the trained model
                                double prediction = svm.svm_predict(model, nodes);
                                Log.d("TrainActivity_DBG", "Prediction: " + prediction);
                            }
                        } else {
                            Log.d("TrainActivity_DBG", "No files found in directory: " + dirPath);
                        }
                    } catch (IOException e) {
                        e.printStackTrace();
                        Log.e("TrainActivity_DBG", "Error loading test image: " + e.getMessage());
                    }

                } else {
                    Log.d("TrainActivity_DBG", "AssetManager is null.");
                }
            }
        });


    }


}
