import os
import json
import numpy as np
from datetime import datetime
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet



def run_training(st):
    """Train MobileNetV2 on chest X-ray dataset and save model + history + temperature."""
    st.info("🚀 Training started... This may take a few minutes.")

    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        zoom_range=0.3,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_gen = ImageDataGenerator(rescale=1./255)

    train_data = train_gen.flow_from_directory(
        'dataset/chest_xray/train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )
    val_data = val_gen.flow_from_directory(
        'dataset/chest_xray/test',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )

    # Base model
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers[:-15]:
        layer.trainable = False
    for layer in base_model.layers[-15:]:
        layer.trainable = True

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.7),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    early_stop = EarlyStopping(patience=5, restore_best_weights=True)
    class_weights = {0: 3.0, 1: 1.0}  # counteract ~74% Pneumonia bias

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=15,
        class_weight=class_weights,
        callbacks=[early_stop]
    )

    
    val_gen_cal = ImageDataGenerator(rescale=1./255)
    val_data_cal = val_gen_cal.flow_from_directory(
        'dataset/chest_xray/val',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        shuffle=False
    )

    raw_preds   = model.predict(val_data_cal).flatten()
    true_labels = val_data_cal.classes

    best_T, best_nll = 1.0, float('inf')
    for T in np.arange(0.5, 5.0, 0.1):
        probs  = np.clip(raw_preds, 1e-6, 1 - 1e-6)
        logits = np.log(probs / (1 - probs))
        cal    = 1 / (1 + np.exp(-logits / T))
        nll    = -np.mean(
            true_labels * np.log(cal + 1e-9) +
            (1 - true_labels) * np.log(1 - cal + 1e-9)
        )
        if nll < best_nll:
            best_nll, best_T = nll, T

    np.save("temperature.npy", best_T)
    st.success(f"✅ Calibration done. Temperature = {best_T:.2f}")

    model.save("pneumonia_model.h5")
    with open("history.json", "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f)

    st.success("✅ Training complete. Model saved as pneumonia_model.h5")
    st.rerun()



def is_xray(uploaded_file) -> tuple[bool, str]:
    """
    Returns (True, 'OK') if the image looks like a chest X-ray.
    Returns (False, reason) otherwise.

    Checks:
      1. Near-grayscale  — X-rays are almost monochrome (R≈G≈B)
      2. Dark background — X-rays have significant dark regions
      3. Not blank       — must have meaningful pixel variance
      4. Not a photo     — natural photos have very high variance
    """
    img = Image.open(uploaded_file).convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    
    avg_color_diff = (
        np.mean(np.abs(R - G)) +
        np.mean(np.abs(R - B)) +
        np.mean(np.abs(G - B))
    ) / 3
    if avg_color_diff > 18:
        return False, (
            f"Image appears to be a color photo (color deviation={avg_color_diff:.1f}). "
            "Please upload a grayscale chest X-ray."
        )

    gray = np.mean(arr, axis=2)


    dark_ratio = np.mean(gray < 85)
    if dark_ratio < 0.25:
        return False, "Image is too bright overall. Chest X-rays typically have a dark background."

    
    std_dev = np.std(gray)
    if std_dev < 15:
        return False, "Image has no meaningful detail. Please upload a real chest X-ray."

    
    if std_dev > 110:
        return False, "Image looks like a natural photograph, not a medical X-ray."

    uploaded_file.seek(0)
    return True, "OK"




def calibrate(prob: float, T: float) -> float:
    """Apply temperature scaling to a raw sigmoid probability."""
    prob = np.clip(prob, 1e-6, 1 - 1e-6)
    logit = np.log(prob / (1 - prob))
    return float(1 / (1 + np.exp(-logit / T)))


def load_pneumonia_model():
    """Load saved model if it exists. Returns (model, loaded: bool)."""
    if os.path.exists("pneumonia_model.h5"):
        return load_model("pneumonia_model.h5"), True
    return None, False


def load_temperature() -> float:
    """Load calibration temperature. Returns 1.0 if not found."""
    try:
        return float(np.load("temperature.npy"))
    except Exception:
        return 1.0


def predict(model, uploaded_file, n_runs: int = 5) -> float:
    """Run model n_runs times and return mean raw probability."""
    uploaded_file.seek(0)
    img       = load_img(uploaded_file, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = [model.predict(img_array)[0][0] for _ in range(n_runs)]
    return float(np.mean(preds))




def create_pdf(name, age, gender, physician, normal_prob, pneumonia_prob, result) -> str:
    file_path = "report.pdf"
    doc       = SimpleDocTemplate(file_path)
    styles    = getSampleStyleSheet()
    content   = []

    content.append(Paragraph("PNEUMONIA DETECTION REPORT", styles["Title"]))
    content.append(Spacer(1, 10))
    content.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    content.append(Spacer(1, 10))
    content.append(Paragraph(f"Patient Name: {name}",    styles["Normal"]))
    content.append(Paragraph(f"Age: {age}",              styles["Normal"]))
    content.append(Paragraph(f"Gender: {gender}",        styles["Normal"]))
    content.append(Paragraph(f"Physician: {physician}",  styles["Normal"]))
    content.append(Spacer(1, 10))
    content.append(Paragraph(f"Result: {result}",                                 styles["Normal"]))
    content.append(Paragraph(f"Normal Probability: {normal_prob*100:.2f}%",       styles["Normal"]))
    content.append(Paragraph(f"Pneumonia Probability: {pneumonia_prob*100:.2f}%", styles["Normal"]))
    content.append(Spacer(1, 10))
    content.append(Paragraph("Note: AI-generated result. Not a medical diagnosis.", styles["Normal"]))

    doc.build(content)
    return file_path



if __name__ == "__main__":
    import sys

    class FakeSt:
        def info(self, msg):    print(f"[INFO] {msg}")
        def success(self, msg): print(f"[OK]   {msg}")
        def warning(self, msg): print(f"[WARN] {msg}")
        def error(self, msg):   print(f"[ERR]  {msg}")
        def rerun(self):        sys.exit(0)

    print("=" * 50)
    print("  Pneumonia Model Trainer")
    print("=" * 50)

    run_training(FakeSt())