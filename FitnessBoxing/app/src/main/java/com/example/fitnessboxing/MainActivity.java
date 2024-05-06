package com.example.fitnessboxing;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.media.MediaPlayer;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.util.Log;

import android.view.animation.Animation;
import android.view.animation.AnimationUtils;


public class MainActivity extends AppCompatActivity {

    private Button exitButton;
    private Button startGameButton; // New button to start the game
    private Button startTrainButton;
    private MediaPlayer backgroundMusic; // MediaPlayer object for background music
    private View squareView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED){
            ActivityCompat.requestPermissions(this, new String[] {Manifest.permission.CAMERA}, 1);}

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        //        squareView = findViewById(R.id.squareView);

        // Initialize buttons
        exitButton = findViewById(R.id.exit);
        startGameButton = findViewById(R.id.start_game); // Correctly initialize the start game button
        startTrainButton = findViewById(R.id.train); // Correctly initialize the start game button



        // Set click listeners
        exitButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                exitApp();
            }
        });

        // Set click listener for the start game button
        startGameButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                startGame();
//                startTrain();
//                startAnimation();
            }

        });

        startTrainButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                startTrain();
//                startAnimation();
            }

        });

        // Initialize background music
        backgroundMusic = MediaPlayer.create(this, R.raw.music_amv_take_it);
        backgroundMusic.setLooping(true); // Set looping to true to repeat the music
        backgroundMusic.start(); // Start playing the background music
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        // Release the MediaPlayer resources
        if (backgroundMusic != null) {
            backgroundMusic.release();
        }
    }

    private void startAnimation() {
        Animation animation = AnimationUtils.loadAnimation(this, R.anim.shrink_animation);
        squareView.startAnimation(animation);
    }

    // Method to exit the app
    private void exitApp() {
//        finish(); // Close the activity
        finishAffinity();
        System.exit(0);
    }

    // Method to start the game activity
    private void startGame() {
        Log.d("MainActivity", "start game");
        Intent intent = new Intent(MainActivity.this, GameActivity.class);
        Log.d("MainActivity", "new intent");
        startActivity(intent);
        Log.d("MainActivity", "start GameActivity");
        backgroundMusic.pause();
    }


    private void startTrain(){
        Log.d("startTrain", "start training");
        Intent trainIntent = new Intent(MainActivity.this, TrainActivity.class);
        Log.d("startTrain", "start TrainActivity");
        startActivity(trainIntent);
        backgroundMusic.pause();
    }
}
