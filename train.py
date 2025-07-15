import tensorflow as tf
import os

model = tf.keras.models.load_model("trained_plant_disease_model.keras")

# Save with full path and confirmation
save_path = r"C:\Users\Md Asif Khan\OneDrive\Desktop\ML Project\New Plant Diseases Dataset(Augmented)\trained_plant_disease_model.h5"
model.save(save_path)
print(f"✅ Model saved at: {save_path}")
