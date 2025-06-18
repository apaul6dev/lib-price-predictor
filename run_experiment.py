from vehicle_price_predictor import (
    Preprocessor,
    test_model,
    evaluate,
    save_model,
    load_model
)
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import pandas as pd

# === 1. Preparar datos ===
print("🔄 Generando datos de ejemplo...")
X, y = make_regression(n_samples=200, n_features=6, noise=5.0, random_state=42)
X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

# === 2. Preprocesamiento ===
print("🧼 Preprocesando datos...")
pre = Preprocessor()
X_clean = pre.clean_data(X)
X_scaled = pre.transform(X_clean)

# === 3. Train/Test split ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === 4. Entrenamiento ===
print("🎯 Entrenando modelo Random Forest...")
y_pred = test_model("random_forest", X_train, y_train, X_test, y_test)

# === 5. Evaluación ===
print("📊 Evaluando resultados...")
metrics = evaluate(y_test, y_pred)
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# === 6. Guardar modelo entrenado ===
print("💾 Guardando modelo...")
from vehicle_price_predictor.models.random_forest.model import RandomForestModel
model = RandomForestModel()
model.train(X_train, y_train)
save_model(model, "random_forest_model.pkl")

# === 7. Cargar modelo y volver a predecir (opcional) ===
print("🔁 Cargando modelo y validando predicción...")
loaded_model = load_model("random_forest_model.pkl")
new_preds = loaded_model.predict(X_test[:5])
print("Predicciones nuevas:", new_preds)
