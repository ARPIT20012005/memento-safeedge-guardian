import cv2
import tensorflow as tf
import numpy as np

# ----------------------------
# LOAD TRAINED MODEL
# ----------------------------
model = tf.keras.models.load_model("child_detection_model.h5")

IMG_SIZE = 96
THRESHOLD = 0.6   # confidence threshold

# ----------------------------
# START WEBCAM
# ----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Webcam not detected")
    exit()

print("✅ Webcam started")

# ----------------------------
# REAL-TIME LOOP
# ----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)[0][0]

    if prediction > THRESHOLD:
        label = "CHILD DETECTED"
        color = (0, 0, 255)  # Red
    else:
        label = "NO CHILD"
        color = (0, 255, 0)  # Green

    # Display text
    cv2.putText(frame, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Child Detection Webcam", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ----------------------------
# CLEANUP
# ----------------------------
cap.release()
cv2.destroyAllWindows()
