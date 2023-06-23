import os
import keras.models

from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import MeanIoU
from  load_data import load_data

def train_model():
    training_path = "drive/MyDrive/Dataset/Training/"
    val_path = "drive/MyDrive/Dataset/Val/"
    image_size = (256, 256)
    batch_size = 32

    training_data = load_data(training_path, image_size, batch_size)
    val_data = load_data(val_path, image_size, batch_size)

    size_data = len(os.listdir("drive/MyDrive/Dataset/Training/Hairs"))
    size_data_val = len(os.listdir("drive/MyDrive/Dataset/Val/Hairs"))

    epochs = 100
    steps_per_epoch = size_data / batch_size
    steps_per_epoch_v = size_data_val / batch_size

    model = keras.models.load_model("drive/MyDrive/model4.h5")
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy', MeanIoU(num_classes=2)])

    checkpoint = ModelCheckpoint('drive/MyDrive/model5.h5', monitor='val_loss', save_best_only=True)

    history = model.fit(training_data,
                    batch_size=batch_size, epochs=epochs,
                    steps_per_epoch= steps_per_epoch,
                    validation_data= val_data,
                    validation_steps=steps_per_epoch_v,
                    callbacks=[checkpoint])