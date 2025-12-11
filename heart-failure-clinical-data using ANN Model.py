import kagglehub, os, pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models

# STEP 2: Download & Load
path = kagglehub.dataset_download("andrewmvd/heart-failure-clinical-data")
print("✅ Dataset Path:", path)
csv_path = os.path.join(path, "heart_failure_clinical_records_dataset.csv")
df = pd.read_csv(csv_path)

print("\n=== Dataset Preview ===")
print(df.head())
print("\nShape:", df.shape)

# STEP 3: Features/Labels
# Target column: DEATH_EVENT (0 = survived, 1 = died)
y = df["DEATH_EVENT"].astype("int32").values
X = df.drop(columns=["DEATH_EVENT"]).astype("float32").values

# STEP 4: Train/Test Split + Scale
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=11, stratify=y)
sc = StandardScaler()
Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)

# STEP 5: ANN
model = models.Sequential([
    layers.Input(shape=(Xtr.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# STEP 6: Train
history = model.fit(Xtr, ytr, epochs=35, batch_size=32, validation_split=0.2, verbose=0)

# STEP 7: Evaluate
probs = model.predict(Xte, verbose=0).ravel()
preds = (probs > 0.5).astype(int)
print(f"\n=== FINAL TEST ACCURACY: {accuracy_score(yte, preds):.4f} ===")
print("\nClassification Report:\n", classification_report(yte, preds, digits=4))

# STEP 8: Plots
plt.figure(); plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Training vs Validation Accuracy — Heart Failure')
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

plt.figure(); plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training vs Validation Loss — Heart Failure')
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

cm = confusion_matrix(yte, preds)
plt.figure(); plt.imshow(cm, cmap='Blues'); plt.title('Confusion Matrix — Heart Failure')
plt.xlabel('Predicted'); plt.ylabel('True'); plt.colorbar()
ticks = np.arange(2); labels = ['Survived (0)','Died (1)']
plt.xticks(ticks, labels, rotation=45, ha='right'); plt.yticks(ticks, labels)
th = cm.max()/2
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i,j]), ha='center', va='center',
                 color='white' if cm[i,j]>th else 'black')
plt.tight_layout(); plt.show()



















