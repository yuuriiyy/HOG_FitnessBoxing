package com.example.fitnessboxing;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.content.res.Resources;
import android.media.MediaPlayer;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;


import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Vector;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.MatOfFloat;
import org.opencv.core.Size;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.HOGDescriptor;

import com.example.fitnessboxing.HandleData;

import libsvm.*;
import libsvm.svm;
import libsvm.svm_node;


//THIS IS THE MOST PRESENT VERSION TBIS IS THE MOST PRESENT VERSIN
public class GameActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2{
    private CameraBridgeViewBase mOpenCvCameraView;
    private CascadeClassifier cascadeClassifier;
    private HOGDescriptor hog;
    // Camera size
    private int myWidth;
    private int myHeight;

    // Mat to store RGBA and Grayscale camera preview frame
    private Mat mRgba;
    private Mat transpose;
    private Mat mGray;
    public int scoreTimeIndex;

    public ArrayList<svm_model> svmModels;
    public svm_model model_l;
    public svm_model model_r;


    private TextView totalScore;
    public Integer currentScore;
    private Button stateButton;
    private MediaPlayer mediaPlayer;
    private ImageView punchImageView;
    private ImageView nextPunchImageView;

    int totalDuration;
//    Vector<PunchData> punchDataVector = HandleData.readCSVFile("../res/raw/data.csv");

    Vector<PunchData> punchDataVector;
    public int punchDataIndex = 0;
    public int lastTime;
    public int lastTimeIndex;


    @Override
    protected void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d("onResume", "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
        } else {
            Log.d("onResume", "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i("base loader", "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    @Override
    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mGray = new Mat(height, width, CvType.CV_8UC1);
        transpose = new Mat(width, height, CvType.CV_8UC4);
        myWidth = width;
        myHeight = height;

    }

    @Override
    public void onCameraViewStopped() {
        mRgba.release();
        mGray.release();
        transpose.release();
    }


    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        mRgba = inputFrame.rgba();
        // Log.d("onCameraFrame", "myWidth" + myWidth); //176
        // Log.d("onCameraFrame", "myHeight" + myHeight); //144

        // Rotate the Camera
        Mat rotImage = Imgproc.getRotationMatrix2D(new Point(mRgba.cols() / 2, mRgba.rows() / 2), 90, 1.2);
        Imgproc.warpAffine(mRgba, mRgba, rotImage, mRgba.size());

        // Flip the image horizontally
        flipImage(mRgba);



        MatOfRect pedestrians = new MatOfRect();
        MatOfDouble foundWeights = new MatOfDouble();
        Mat grayFrame = new Mat();


        // Convert image to grayscale
        Imgproc.cvtColor(mRgba, grayFrame, Imgproc.COLOR_RGBA2GRAY);
        Imgproc.equalizeHist(grayFrame, grayFrame);


        // Applying HOG detector for pedestrians
        detectPedestrians(grayFrame);



        return mRgba;
    }

    private void flipImage(Mat image) {
        int rows = image.rows();
        int cols = image.cols();
        int channels = image.channels();
        byte[] data = new byte[rows * cols * channels];
        image.get(0, 0, data);
        byte[] newData = new byte[rows * cols * channels];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                for (int k = 0; k < channels; k++) {
                    int index = (i * cols + j) * channels + k;
                    int flippedIndex = (i * cols + (cols - 1 - j)) * channels + k;
                    newData[flippedIndex] = data[index];
                }
            }
        }
        image.put(0, 0, newData);
    }

    private void detectPedestrians(Mat grayFrame) {
        MatOfRect pedestrians = new MatOfRect();
        MatOfDouble foundWeights = new MatOfDouble();

        // Detect pedestrians
        hog.detectMultiScale(grayFrame, pedestrians, foundWeights);

        // Draw rectangles around detected pedestrians
        Rect[] pedestriansArray = pedestrians.toArray();
        if (pedestriansArray.length > 0) {
//            for (Rect rect : pedestriansArray) {
            Rect rect = pedestriansArray[0];
            Imgproc.rectangle(mRgba, rect.tl(), rect.br(), new Scalar(0, 255, 0), 3);
            // Extract HoG features for the rect area
            Mat pedestrianRegion = grayFrame.submat(rect);
            HoGFeature hoGFeatureExtractor = new HoGFeature(pedestrianRegion);
            double[] hogFeatures = hoGFeatureExtractor.extractFeatures();
            Log.i("GameActivity", "Pedestrian Detection: " + hogFeatures.length);

            svm_node[] nodes = new svm_node[hogFeatures.length];
            for (int i = 0; i < hogFeatures.length; i++) {
                svm_node node = new svm_node();
                node.index = i + 1;
                node.value = hogFeatures[i];
                nodes[i] = node;
            }



            if(punchDataVector.get(punchDataIndex).punch == 1) {
//                double prediction_l = svm.svm_predict(model_l, nodes);
                Log.i("GameActivity", "Pedestrian Prediction_l: testing" );
//                Log.i("GameActivity", "Pedestrian Prediction_l: " + prediction_l);
//                if((int)prediction_l == 1){
//                    currentScore += 100;
//                }
            }
            else {
                double prediction_r = svm.svm_predict(model_r, nodes);
                Log.i("GameActivity", "Pedestrian Prediction_r: " + prediction_r);
                if ((int) prediction_r == 1) {
                    currentScore += 100;
                }
            }

//            }
        }
    }



    //new
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        if (!OpenCVLoader.initDebug()) {
            Log.e(this.getClass().getSimpleName(), "  OpenCVLoader.initDebug(), not working.");
        } else {
            Log.d(this.getClass().getSimpleName(), "  OpenCVLoader.initDebug(), working.");
        }

        hog = new HOGDescriptor();
        hog.setSVMDetector(HOGDescriptor.getDefaultPeopleDetector());


        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_game);

        totalScore = findViewById(R.id.score);
        stateButton = findViewById(R.id.state);
        punchImageView = findViewById(R.id.current_punch);
        nextPunchImageView = findViewById(R.id.next_punch);
        Button punchButton = findViewById(R.id.bottom1);
        currentScore = 0;
        scoreTimeIndex = 0;
        AssetManager assetManager = this.getAssets();
//        model_l = SVMTrainer.main(null, assetManager, "left");
        model_r = SVMTrainer.main(null, assetManager, "right");


        try {
            InputStream inputStream = getResources().openRawResource(R.raw.data);
            Log.d("File Check", "CSV file exists in res/raw");
        }
        catch (Resources.NotFoundException e) {
            Log.d("File Check", "CSV file does not exist in res/raw");
        }

        punchDataVector = HandleData.readCSVFile(getApplicationContext().getResources().openRawResource(R.raw.data));

        Log.d("Punch Data", "size: " + punchDataVector.size());

        if (!punchDataVector.isEmpty()) {
            lastTimeIndex = punchDataVector.size() - 1;
            lastTime = punchDataVector.get(lastTimeIndex).time;
        }

        for (PunchData punchData : punchDataVector) {
            int time = punchData.time;
            int punch = punchData.punch;
            Log.d("Punch Data", "Time: " + time + ", Punch: " + punch);

        }


        // Initialize the MediaPlayer with the music file
        mediaPlayer = MediaPlayer.create(this, R.raw.music_smart);
        totalDuration = mediaPlayer.getDuration();
        int totalMinutes = totalDuration / 1000 / 60;
        int totalSeconds = (totalDuration / 1000) % 60;
        Log.d("Music Time", "Total Time: " + totalMinutes + ":" + totalSeconds);

        playMusic();

        stateButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (mediaPlayer.isPlaying()) {
                    pauseMusic();
                } else {
                    playMusic();
                }
            }
        });

        punchButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                changePunch();
            }
        });
        // Start a thread to update the current and total music playing time
        new Thread(new Runnable() {
            @Override
            public void run() {
                while (true) {
                    try {
                        Thread.sleep(1000); // Update every second
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }

                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            updateMusicTime();
                            totalScore.setText(String.valueOf(currentScore));
                        }
                    });
                }
            }
        }).start();



        // Set a listener to trigger when the MediaPlayer finishes playing the music
        mediaPlayer.setOnCompletionListener(new MediaPlayer.OnCompletionListener() {
            @Override
            public void onCompletion(MediaPlayer mp) {
                // Start the ResultActivity when the music finishes
                Intent intent = new Intent(GameActivity.this, ResultActivity.class);
                intent.putExtra("score", currentScore);
                startActivity(intent);
            }
        });
        // Setup OpenCV Camera View
        if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED){

            mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.opencv_camera_preview);
            // Use main camera with 0 or front camera with 1
            mOpenCvCameraView.setCameraIndex(1);
            // Force camera resolution, ignored since OpenCV automatically select best ones
            //         mOpenCvCameraView.setMaxFrameSize(720, 1280);
            mOpenCvCameraView.setCvCameraViewListener(this);}



    }
    private void updateMusicTime() {
        if (mediaPlayer.isPlaying()) {
            int currentPosition = mediaPlayer.getCurrentPosition();
            Log.d("Music Time", "currentPosition" + currentPosition);

            // Convert milliseconds to minutes and seconds
            int currentMinutes = currentPosition / 1000 / 60;
            int currentSeconds = (currentPosition / 1000) % 60;
            int currentTimeIndex = (currentPosition / 1000);
//            Log.d("Music Time", "Current Time Index: " + currentTimeIndex);
//            Log.d("Music Time", "Current Time: " + currentMinutes + ":" + currentSeconds);
            if(punchDataIndex <= lastTimeIndex && punchDataVector.get(punchDataIndex).time  == currentTimeIndex){
                if(punchDataVector.get(punchDataIndex).punch == 1){
//                    Log.d("From Vector", "Punch: Right");
                    punchImageView.setImageResource(R.drawable.right_punch);
                    punchImageView.setTag("res/drawable/right_punch.png"); // Update the tag to the new image resource ID
                    Log.d("current Punch", "punch changed");
                }
                else{
//                    Log.d("From Vector", "Punch: Left");
                    punchImageView.setImageResource(R.drawable.left_punch);
                    punchImageView.setTag("res/drawable/left_punch.png"); // Update the tag to the new image resource ID
                    Log.d("current Punch", "punch changed");
                }

                if(punchDataIndex + 1 <= lastTimeIndex){
                    if(punchDataVector.get(punchDataIndex + 1).punch == 1) {
                        nextPunchImageView.setImageResource(R.drawable.right_punch_next);
                        nextPunchImageView.setTag("res/drawable/right_punch_next.png"); // Update the tag to the new image resource ID
                    }
                    else{
                        nextPunchImageView.setImageResource(R.drawable.left_punch_next);
                        nextPunchImageView.setTag("res/drawable/left_punch_next.png"); // Update the tag to the new image resource ID
                    }
                }
                else{
                    nextPunchImageView.setImageResource(R.drawable.end);
                    nextPunchImageView.setTag("res/drawable/end.png");
                }

                punchDataIndex += 1;
            }

        }
    }

    private void changePunch(){
        // Get the current drawable resource ID of the punch image
        String currentPunchImageResource = (String) punchImageView.getTag();
        Log.d("click PUNCH!", "currentPunchImageResource: " + currentPunchImageResource);

        // Toggle between left and right punch images
        if (Objects.equals(currentPunchImageResource, "res/drawable/left_punch.png")) {
            Log.d("current Punch", "left_punch --> right_punch");
            punchImageView.setImageResource(R.drawable.right_punch);
            punchImageView.setTag("res/drawable/right_punch.png"); // Update the tag to the new image resource ID
            Log.d("current Punch", "punch changed");
        } else if (Objects.equals(currentPunchImageResource, "res/drawable/right_punch.png")) {
            Log.d("current Punch", "right_punch --> left_punch");
            punchImageView.setImageResource(R.drawable.left_punch);
            punchImageView.setTag("res/drawable/left_punch.png"); // Update the tag to the new image resource ID
            Log.d("current Punch", "punch changed");
        }
    }


    private void animation(){
        // animationTime = punchDataVector.get(punchDataIndex).time - 2;
        // if currentTimeIndex == animationTime:
        // create a shrink animation
    }

    private void playMusic() {
        stateButton.setText("||"); // Change button text to pause symbol
        mediaPlayer.start(); // Start playing the music
    }

    private void pauseMusic() {
        stateButton.setText("▶"); // Change button text to play symbol
        mediaPlayer.pause(); // Pause the music
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
        if (mediaPlayer != null) {
            mediaPlayer.release(); // Release the MediaPlayer resources
        }
    }
}
