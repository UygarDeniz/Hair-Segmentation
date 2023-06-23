from tensorflow.keras.preprocessing.image import ImageDataGenerator
def load_data(path, img_size, batch_size):
    data_gen = ImageDataGenerator(rotation_range=0.2,
                         rescale = 1./255,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True,
                         fill_mode='nearest')

    image_generator = data_gen.flow_from_directory(
        path,
        classes = ["Hairs"],
        target_size=img_size,
        color_mode = "rgb",
        batch_size = batch_size,
        class_mode=None,
        seed=1
    )

    mask_generator = data_gen.flow_from_directory(
        path,
        classes = ["Masks"],
        target_size=img_size,
        class_mode=None,
        batch_size = batch_size,
        color_mode='grayscale',
        seed=1
    )

    data_generator = zip(image_generator, mask_generator)
    return data_generator