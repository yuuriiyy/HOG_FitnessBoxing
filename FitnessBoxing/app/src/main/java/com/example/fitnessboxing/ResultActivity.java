package com.example.fitnessboxing;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.TextView;
import android.media.MediaPlayer;
import android.widget.Button;
import androidx.appcompat.app.AppCompatActivity;

public class ResultActivity extends AppCompatActivity {

    private TextView resultScoreTextView;
    private TextView resultBestTimesTextView;
    private TextView resultGoodTimesTextView;
    private TextView resultMissTimesTextView;
    private MediaPlayer resultMusic; // MediaPlayer object for background music


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_result);
        Button homeButton = findViewById(R.id.home_button);

        // Set an OnClickListener for the "home" button
        homeButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // Create an Intent to navigate back to MainActivity
                Intent intent = new Intent(ResultActivity.this, MainActivity.class);
                startActivity(intent); // Start the MainActivity
            }
        });

        // Initialize TextViews
        resultScoreTextView = findViewById(R.id.result_score);
        resultBestTimesTextView = findViewById(R.id.result_bestTimes);
        resultGoodTimesTextView = findViewById(R.id.result_goodTimes);
        resultMissTimesTextView = findViewById(R.id.result_missTimes);

        // Example values (you can replace these with your actual values)
        Intent intent = getIntent();
        int score = intent.getIntExtra("score", 0);
        int bestTimes = 10;
        int goodTimes = 20;
        int missTimes = 5;

        // Set values to TextViews
        resultScoreTextView.setText(getString(R.string.result_score, score));
        resultBestTimesTextView.setText(getString(R.string.result_bestTimes, bestTimes));
        resultGoodTimesTextView.setText(getString(R.string.result_goodTimes, goodTimes));
        resultMissTimesTextView.setText(getString(R.string.result_missTimes, missTimes));
    }
}
