import numpy as np
from tensorflow.keras.preprocessing.image import load_img

def prepare_images(df, dataset_path, img_size=(128, 128)):
    x_data = []
    for file in df.image:
        img = load_img(f"{dataset_path}/{file}", target_size=img_size)
        img = img.convert('L')  # grayscale
        img = np.array(img)
        x_data.append(img)

    x_data = np.array(x_data).reshape(-1, img_size[0], img_size[1], 1) / 255.0
    return x_data, np.array(df.gender), np.array(df.age)