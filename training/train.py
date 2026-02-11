import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

#  Load Data
df = pd.read_csv("data/server_metrics.csv")

print("Dataset Loaded:")
print(df.head())

# Separate Features & Label
X = df.drop("crash", axis=1)
y = df["crash"]

print("\nFeatures shape:", X.shape)
print("Label shape:", y.shape)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

#  Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 5️⃣ Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy:.2f}")

# 6️⃣ Save Model
joblib.dump(model, "model/model.pkl")
print("\n✅ Model saved as model/model.pkl")
