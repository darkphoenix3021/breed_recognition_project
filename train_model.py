import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import os

# --- SETUP AND DATA PREPARATION ---
IMG_HEIGHT, IMG_WIDTH = 224, 224
DATASET_PATH = 'dataset/'

# Check if dataset directory exists
if not os.path.exists(DATASET_PATH) or not os.listdir(DATASET_PATH):
    print(f"Error: The 'dataset' folder is empty or does not exist.")
    print("Please create it and add subfolders for each breed with images inside.")
else:
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_generator = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    # --- BUILD, TRAIN, AND SAVE ---
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(train_generator.num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print("Starting model training...")
    model.fit(train_generator, epochs=5) 

    print("Training complete. Saving model and labels...")
    model.save('cattle_breed_classifier.h5') # This line creates the model file

    labels = list(train_generator.class_indices.keys())
    with open('class_labels.txt', 'w') as f: # This section creates the text file
        for label in labels:
            f.write(f"{label}\n")

    print("Successfully created 'cattle_breed_classifier.h5' and 'class_labels.txt'!")
