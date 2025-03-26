import numpy as np
import matplotlib.pyplot as plt

gender_dict = {0: "Male", 1: "Female"}

def predict_and_display(model, x_data, y_gender, y_age, index):
    pred = model.predict(x_data[index].reshape(1, 128, 128, 1))
    pred_gender = gender_dict[round(pred[0][0][0])]
    pred_age = round(pred[1][0][0])

    print(f"Original: Gender = {gender_dict[y_gender[index]]}, Age = {y_age[index]}")
    print(f"Predicted: Gender = {pred_gender}, Age = {pred_age}")
    plt.imshow(x_data[index].reshape(128, 128), cmap='gray')
    plt.axis('off')
    plt.show()