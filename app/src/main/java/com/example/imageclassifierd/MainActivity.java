package com.example.imageclassifierd;

import static com.example.imageclassifierd.R.*;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.imageclassifierd.ml.ModelUnquant;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {

    TextView result;
    ImageView imageView;
    Button picture, gallery;
    int imageSize = 224;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        result = findViewById(R.id.result);
        imageView = findViewById(R.id.imageView);
        picture = findViewById(R.id.button);
        gallery = findViewById(R.id.button_gallery);

        picture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Launch camera if we have permission
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 1);
                } else {
                    // Request camera permission if we don't have it.
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }
            }
        });

        gallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Launch gallery to pick an image
                Intent galleryIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(galleryIntent, 2);
            }
        });
    }

    public void classifyImage(Bitmap image) {
        try {
            ModelUnquant model = ModelUnquant.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
            int pixel = 0;
            for (int i = 0; i < imageSize; i++) {
                for (int j = 0; j < imageSize; j++) {
                    int val = intValues[pixel++]; // RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255.f));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            ModelUnquant.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            int maxPos = 0;
            float maxConfidence = 0;
            for (int i = 0; i < confidences.length; i++) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }

            String[] classes = {"Banana Yellow_BlackSigatoka", "Banana Cordana", "Banana Healthy", "Banana Insect Pest",
                    "Banana Moko", "Banana Mosaic", "Banana Panama", "Coconut Bud Root Dropping",
                    "Coconut Bud Rot", "Coconut Catterpillar", "Coconut Catterpillars Spot",
                    "Coconut Grayleaf Spot", "Coconut Leaf Rot", "Coconut Stem Bleeding", "Coconut Fruit Healthy",
                    "Coconut Fruit Infected", "Corn Cercospora Leaf Spot", "Corn Common Rust",
                    "Corn Healthy", "Corn Northern Leaf Blight", "Pineapple Crown Rot",
                    "Pineapple Fruit Fasciation", "Pineapple Fruit Rot", "Pineapple Healthy",
                    "Pineapple Mealybug Wilt", "Pineapple Multiple Crop Disorder", "Pineapple Root Rot",
                    "Sugarcane Healthy", "Sugarcane Mosaic Virus", "Sugarcane Red Rot",
                    "Sugarcane Rust", "Sugarcane Yellow"};

            // Display only the top prediction
            result.setText(String.format("Prediction: %s (Confidence: %.1f%%)", classes[maxPos], maxConfidence * 100));

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // Handle the exception
            e.printStackTrace();
        }
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if (resultCode == RESULT_OK && data != null) {
            Bitmap image = null;
            if (requestCode == 1) {
                // Handle camera image
                image = (Bitmap) data.getExtras().get("data");
            } else if (requestCode == 2) {
                // Handle gallery image
                Uri imageUri = data.getData();
                try {
                    InputStream imageStream = getContentResolver().openInputStream(imageUri);
                    image = BitmapFactory.decodeStream(imageStream);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }

            if (image != null) {
                int dimension = Math.min(image.getWidth(), image.getHeight());
                image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
                imageView.setImageBitmap(image);

                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                classifyImage(image);
            }
        }
        super.onActivityResult(requestCode, resultCode, data);
    }
}