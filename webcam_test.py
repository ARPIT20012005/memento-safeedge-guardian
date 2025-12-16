import cv2
import tensorflow as tf
import numpy as np

# Load trained model
model = tf.keras.models.load_model("child_detection_model.h5")

IMG_SIZE = 96
THRESHOLD = 0.35

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]
    print(prediction)

    if prediction > THRESHOLD:
        label = "CHILD DETECTED"
        color = (0, 0, 255)
    else:
        label = "NO CHILD"
        color = (0, 255, 0)

    cv2.putText(frame, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Child Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
