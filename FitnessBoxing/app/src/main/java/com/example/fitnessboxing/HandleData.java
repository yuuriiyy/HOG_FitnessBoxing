package com.example.fitnessboxing;



import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Vector;

class PunchData {
    int time;
    int punch;

    public PunchData(int time, int punch) {
        this.time = time;
        this.punch = punch;
    }
}

public class HandleData {
    public static Vector<PunchData> readCSVFile(InputStream inputStream) {
        Vector<PunchData> punchDataVector = new Vector<>();

        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
            String line;

            // Read each line of the CSV file
            while ((line = reader.readLine()) != null) {
                // Split the line into columns using comma as the delimiter
                String[] columns = line.split(",");

                // Parse the values for the "time" and "punch" columns
                int time = Integer.parseInt(columns[0]);
                int punch = Integer.parseInt(columns[1]);

                // Create a PunchData object and add it to the Vector
                punchDataVector.add(new PunchData(time, punch));
            }

            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        return punchDataVector;
    }
}
