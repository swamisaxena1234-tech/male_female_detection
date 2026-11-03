from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np

# -----------------------------
# 1. Load model
# -----------------------------
model = load_model('gender_detection_model.h5')
print("Model Loaded Successfully")

# -----------------------------
# 2. Evaluate Model on Test Set
# -----------------------------
test_dir = 'gender_dataset/test'   # <-- change if needed

test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory(
    test_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

loss, accuracy = model.evaluate(test_set)
print(f"\nTest Loss : {loss:.4f}")
print(f"Test Accuracy : {accuracy * 100:.2f}%")

# -----------------------------
# 3. Predict Single Image
# -----------------------------
img_path = 'images.jpg'  # change this path
img = load_img(img_path, target_size=(64, 64))

img_array = img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

print("\nPrediction Result:")
if prediction[0][0] > 0.5:
    print("Predicted: Female")
else:
    print("Predicted: Male")
