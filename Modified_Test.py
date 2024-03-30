import tkinter as tk
from tkinter import filedialog
import cv2
import os
import numpy as np
from keras.models import load_model

model = load_model('CNN.model')

data_dir = "dataset"
class_names = os.listdir(data_dir)

def predict_image():
    file_path = filedialog.askopenfilename()  # Choose an image using tkinter dialog
    if file_path:
        image = cv2.imread(file_path)
        im = cv2.imread(file_path)
        image = cv2.resize(image, (100, 100))  # Adjust the image size as needed
        image = np.expand_dims(image, axis=0)
        image = image / 255.0  # Normalize the image

        prediction = model.predict(image)
        predicted_class = class_names[np.argmax(prediction)]

        result_label.config(text=f"Predicted Class: {predicted_class}")
        print(predicted_class)

        global contour
        if "Bad" in predicted_class:
            # Filter dark pixels based on color range (adjust as needed)
            dark_mask = cv2.inRange(im, (0, 0, 0), (50, 50, 50))
            dark_pixels = cv2.countNonZero(dark_mask)

            total_pixels = image.shape[1] * image.shape[0]  # Total image pixels
            percentage_dark_pixels = (dark_pixels / total_pixels) 
            print(f"Percentage of dark pixels: {percentage_dark_pixels:.2f}%")

            cv2.imshow("Detected spots", dark_mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

root = tk.Tk()
root.title("Image Classifier")

choose_button = tk.Button(root, text="Choose Image", command=predict_image)
choose_button.pack()

result_label = tk.Label(root, text="")
result_label.pack()

root.mainloop()
