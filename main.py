import cv2
import keras.models
import numpy as np
from change_hair_color import change_hair_color
def predict_mask(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image= image / 255
    height, width = image.shape[:2]
    image = cv2.resize(image, (256, 256))
    image = image.reshape((1,) + image.shape)

    pred = model.predict(image)

    pred = pred.reshape(256,256)
    pred = cv2.resize(pred, (width, height))
    pred = np.where(pred > 0.5, 255, 0).astype(np.uint8)
    return pred
def display_result(result):
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

model = keras.models.load_model("trained_models/model4.h5")


"""""
image_path = "Data/00105.jpg"
image = cv2.imread(image_path)
pred_mask = predict_mask(image, model)
color = (226, 43, 138)
changed_hair = change_hair_color(image.copy(), pred_mask, color, alpha=0.25)

pred_mask= cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)

image = cv2.resize(image, (640, 640))
pred_mask = cv2.resize(pred_mask, (640, 640))
#changed_hair = cv2.resize(changed_hair, (640, 640))

result = np.hstack([image, pred_mask])
display_result(result)
"""""














