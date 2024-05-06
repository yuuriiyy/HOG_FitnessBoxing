package com.example.fitnessboxing;

import android.content.Context;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

public class ModelSaver {

    public static void saveModelToDownloads(Context context, String fileName, byte[] modelData) {
        File downloadsFolder = new File(context.getExternalFilesDir(null), "Download");
        if (!downloadsFolder.exists()) {
            downloadsFolder.mkdirs(); // Create the Downloads folder if it doesn't exist
        }

        File modelFile = new File(downloadsFolder, fileName);

        try {
            FileOutputStream fos = new FileOutputStream(modelFile);
            fos.write(modelData);
            fos.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
