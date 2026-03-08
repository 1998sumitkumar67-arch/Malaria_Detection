import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

dataset_path = r"C:\Work\Malaria_Detection\dataset"

# Training Data Generator (with augmentation)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    shear_range=0.2
)

# Validation & Test (only normalization)
test_datagen = ImageDataGenerator(rescale=1./255)

# Training Data
train_data = train_datagen.flow_from_directory(
    dataset_path + "/train",
    target_size=(224,224),
    batch_size=32,
    class_mode="binary"
)

# Validation Data
validation_data = test_datagen.flow_from_directory(
    dataset_path + "/validation",
    target_size=(224,224),
    batch_size=32,
    class_mode="binary"
)

# Test Data
test_data = test_datagen.flow_from_directory(
    dataset_path + "/test",
    target_size=(224,224),
    batch_size=32,
    class_mode="binary"
)

print("Dataset Loaded Successfully")