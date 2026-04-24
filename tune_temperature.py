import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scipy.optimize import minimize

# Load model
model = load_model("pneumonia_model.h5")

# Validation data (same as your test folder)
val_gen = ImageDataGenerator(rescale=1./255)

val_data = val_gen.flow_from_directory(
    'dataset/chest_xray/test',
    target_size=(224,224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Get predictions
logits = model.predict(val_data)
y_true = val_data.classes

# Convert sigmoid output → logits
def sigmoid_inverse(p):
    p = np.clip(p, 1e-6, 1-1e-6)
    return np.log(p / (1 - p))

logits = sigmoid_inverse(logits)

# Apply temperature scaling
def apply_temp(logits, T):
    return 1 / (1 + np.exp(-logits / T))

# Loss function (NLL)
def nll(T):
    T = T[0]
    probs = apply_temp(logits, T)
    probs = np.clip(probs, 1e-6, 1-1e-6)
    
    loss = -np.mean(
        y_true * np.log(probs) + (1 - y_true) * np.log(1 - probs)
    )
    return loss

# Optimize temperature
opt = minimize(nll, x0=[1.0], bounds=[(0.05, 10)])

T_opt = opt.x[0]
print("Optimal Temperature:", T_opt)

# Save temperature
np.save("temp.npy", T_opt)