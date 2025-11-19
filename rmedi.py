# entrenar_medicamentos.py
import firebase_admin
from firebase_admin import credentials, firestore
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf
import os
import json
from itertools import combinations
from nltk.stem.snowball import SpanishStemmer
import time
import datetime

# Stemmer español
stemmer = SpanishStemmer()

# -----------------------------
# 1️⃣ Inicializar Firebase
# -----------------------------
cred = credentials.Certificate(r"C:\Users\alexa\FirebaseKeys\serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# -----------------------------
# 2️⃣ Extraer medicamentos desde Firestore
# -----------------------------
meds_docs = db.collection("medicamentos").stream()
datos = {"medicamentos": []}

for doc in meds_docs:
    m = doc.to_dict()
    # Aseguramos campos básicos (evita KeyError si faltan campos)
    medicamento = {
        "nombre": m.get("nombre", "").strip(),
        "descripcion": m.get("descripcion", ""),
        "usos": m.get("usos", ""),
        "reacciones": m.get("reacciones", ""),
        "presentacion": m.get("presentacion", ""),
        "contraindicaciones": m.get("contraindicaciones", ""),
        "dosis": m.get("dosis", ""),
        "tipo": m.get("tipo", "")
    }
    datos["medicamentos"].append(medicamento)

# Si no hay medicamentos, terminamos
if not datos["medicamentos"]:
    print("❌ No se encontraron documentos en la colección 'medicamentos'.")
    raise SystemExit

# -----------------------------
# 3️⃣ Normalización y stemming
# -----------------------------
def normalize(text):
    accents = 'áéíóúüÁÉÍÓÚÜ'
    replacements = 'aeiouuAEIOUU'
    for i in range(len(accents)):
        text = text.replace(accents[i], replacements[i])
    return ''.join(c for c in text.lower() if c.isalnum() or c.isspace())

def stem_list(lst):
    return [stemmer.stem(normalize(s)) for s in lst if s and s.strip()]

# Construimos tokens (similar a "sintomas" en tu primer script)
tokens = set()
tokens_por_medicamento = {}

medicamentos_nombres = [m["nombre"] for m in datos["medicamentos"]]

for m in datos["medicamentos"]:
    # Tomamos: nombre dividido en palabras + tipo (por si el usuario escribe 'analgésico' o 'antinflamatorio')
    partes = []
    partes.extend(m["nombre"].split())
    if m.get("tipo"):
        partes.extend(m["tipo"].split())
    # Podríamos incluir palabras importantes de 'presentacion' o 'descripcion' si queremos más robustez:
    # partes.extend(m.get("presentacion", "").split())
    tokens_norm = stem_list(partes)
    tokens_por_medicamento[m["nombre"]] = tokens_norm
    for t in tokens_norm:
        tokens.add(t)

tokens = sorted(list(tokens))

# -----------------------------
# 3.1️⃣ Comparar con versión previa (control)
# -----------------------------
control_file = "control_medicamentos.json"
cambio_detectado = True

if os.path.exists(control_file):
    with open(control_file, "r", encoding="utf-8") as f:
        control_data = json.load(f)

    prev_tokens = []
    prev_meds = []

    for e in control_data:
        prev_tokens.extend(stem_list(e.get("nombre", "").split()) + stem_list([e.get("tipo", "")]))
        prev_meds.append(e.get("nombre", ""))

    # Comparamos tokens únicos y lista de nombres (ordenadas)
    if sorted(list(set(prev_tokens))) == sorted(list(set(tokens))) and sorted(prev_meds) == sorted(medicamentos_nombres):
        cambio_detectado = False

if cambio_detectado:
    print("⚠️ Se detectaron cambios en medicamentos o tokens. Se entrenará la red.")
else:
    print("✅ No hay cambios. No es necesario entrenar.")

# Guardar control actualizado
with open(control_file, "w", encoding="utf-8") as f:
    json.dump(datos["medicamentos"], f, ensure_ascii=False, indent=4)

# -----------------------------
# 4️⃣ Preparar datos (X,y)
# -----------------------------
X = []
y = []

for m in datos["medicamentos"]:
    tokens_norm = tokens_por_medicamento[m["nombre"]]

    # Generamos combinaciones de tokens como ejemplos de entrada (1 a 3 tokens)
    max_r = min(len(tokens_norm), 3) if tokens_norm else 1
    if max_r == 0:
        # Si no hay tokens (nombre vacío), añadimos vector vacío
        vec = [0] * len(tokens)
        X.append(vec)
        salida = [1 if m["nombre"] == nombre else 0 for nombre in medicamentos_nombres]
        y.append(salida)
    else:
        for r in range(1, max_r + 1):
            for combo in combinations(tokens_norm, r):
                vector = [1 if t in combo else 0 for t in tokens]
                X.append(vector)
                salida = [1 if m["nombre"] == nombre else 0 for nombre in medicamentos_nombres]
                y.append(salida)

# Si no hay ejemplos generados por alguna razón, creamos al menos uno por medicamento (precaución)
if not X:
    for m in datos["medicamentos"]:
        X.append([0] * len(tokens))
        y.append([1 if m["nombre"] == nombre else 0 for nombre in medicamentos_nombres])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

print(f"📦 Ejemplos generados: {X.shape[0]} - Dim entrada: {X.shape[1]} - Clases: {y.shape[1]}")

# -----------------------------
# 5️⃣ Definir red (misma estructura que tu código original)
# -----------------------------
model = Sequential([
    Dense(64, input_dim=len(tokens), activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(len(medicamentos_nombres), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# -----------------------------
# 6️⃣ Entrenamiento (solo si hubo cambios)
# -----------------------------
history = None

if cambio_detectado:
    print("⏳ Entrenando red neuronal para medicamentos...")
    # Para evitar overfitting con pocos datos se podría usar más validación; aquí mantenemos simple
    history = model.fit(X, y, epochs=400, verbose=1)
    print("✅ Entrenamiento finalizado.")
else:
    print("⚠️ La red NO fue entrenada porque no hubo cambios.")

# -----------------------------
# 7️⃣ Guardar modelo
# -----------------------------
model.save("modelo_medicamentos.h5")
print("✅ Modelo guardado: modelo_medicamentos.h5")

try:
    print("📦 Convirtiendo a TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open("modelo_medicamentos.tflite", "wb") as f:
        f.write(tflite_model)
    print("✅ Guardado modelo_medicamentos.tflite")
except Exception as e:
    print("❌ Error TFLite:", e)

print("🔍 Verificando integridad del TFLite...")
try:
    interpreter = tf.lite.Interpreter(model_path="modelo_medicamentos.tflite")
    interpreter.allocate_tensors()
    print("✅ Modelo TFLite válido.")
except Exception as e:
    print("❌ Error verificando el modelo TFLite:", e)

# -----------------------------
# 8️⃣ Actualizar version.txt
# -----------------------------
version_file = "version.txt"

def read_version():
    if not os.path.exists(version_file):
        return 0
    try:
        return int(open(version_file, "r").read().strip())
    except:
        return 0

def update_version():
    new_v = read_version() + 1
    with open(version_file, "w") as f:
        f.write(str(new_v))
    print(f"🆙 Versión actualizada a: {new_v}")
    return new_v

if cambio_detectado:
    nueva_version = update_version()
else:
    nueva_version = read_version()

# -----------------------------
# 9️⃣ Generar reporte JSON
# -----------------------------
if cambio_detectado and history is not None:
    total_params = model.count_params()
    final_loss = history.history["loss"][-1]
    final_accuracy = history.history["accuracy"][-1]
    eficiencia = round(final_accuracy * 100, 2)

    reporte = {
        "fecha": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tipo_red": "MLP - Fully Connected",
        "eficiencia_modelo": f"{eficiencia}%",
        "capas": [
            {"tipo": "Dense", "unidades": 64, "activacion": "relu"},
            {"tipo": "Dropout", "rate": 0.2},
            {"tipo": "Dense", "unidades": 64, "activacion": "relu"},
            {"tipo": "Dropout", "rate": 0.2},
            {"tipo": "Dense", "unidades": len(medicamentos_nombres), "activacion": "softmax"}
        ],
        "parametros_totales": total_params,
        "num_tokens": len(tokens),
        "num_medicamentos": len(medicamentos_nombres),
        "epochs_solicitados": 400,
        "epochs_realizados": len(history.history["loss"]),
        "loss_final": float(final_loss),
        "accuracy_final": float(final_accuracy),
        "tokens_generados": tokens,
        "medicamentos_detectados": medicamentos_nombres,
        "version_modelo": nueva_version
    }

    with open("reporte_entrenamiento_medicamentos.json", "w", encoding="utf-8") as f:
        json.dump(reporte, f, ensure_ascii=False, indent=4)

    print("📊 Reporte generado correctamente: reporte_entrenamiento_medicamentos.json")
else:
    print("ℹ️ No se generó reporte porque no hubo entrenamiento.")

# -----------------------------
# 🔟 Función de predicción (devuelve datos completos del medicamento)
# -----------------------------
def predict_medicamento(user_input, top_n=3, threshold=5):
    """
    user_input: texto ingresado por el usuario (nombre o fragmento)
    top_n: cuántos resultados devolver
    threshold: probabilidad mínima (%) para considerar
    """
    input_words = stem_list(user_input.split())
    input_vector = [0] * len(tokens)

    for i, t in enumerate(tokens):
        for w in input_words:
            # coincidencia por substring/stem
            if w in t or t in w:
                input_vector[i] = 1
                break

    input_vector = np.array([input_vector], dtype=np.float32)
    predictions = model.predict(input_vector)[0]

    results = []
    for i, prob in enumerate(predictions):
        prob_pct = float(prob) * 100.0
        if prob_pct >= threshold:
            med_nombre = medicamentos_nombres[i]
            # Buscar info completa del medicamento en 'datos'
            med_info = next((m for m in datos["medicamentos"] if m["nombre"] == med_nombre), None)
            results.append({
                "medicamento": med_nombre,
                "probabilidad": round(prob_pct, 1),
                "info": med_info
            })

    # Ordenar por probabilidad descendente y devolver top_n
    return sorted(results, key=lambda x: x["probabilidad"], reverse=True)[:top_n]

# -----------------------------
# 🧪 Ejemplo de uso
# -----------------------------
if __name__ == "__main__":
    # Ejemplo: el usuario escribe el nombre o parte del nombre
    ejemplo_usuario = "ibuprofeno"   # Cambia esto para probar otros inputs
    print(f"\n🔎 Buscando para: '{ejemplo_usuario}'")
    resultado = predict_medicamento(ejemplo_usuario, top_n=5, threshold=1)  # threshold bajo para mostrar posibilidades
    if not resultado:
        print("❌ No se encontraron coincidencias con la probabilidad solicitada.")
    else:
        for r in resultado:
            print(f"\nMedicamento: {r['medicamento']} - Probabilidad: {r['probabilidad']}%")
            info = r.get("info", {})
            if info:
                print("  Descripción:", info.get("descripcion", ""))
                print("  Usos:", info.get("usos", ""))
                print("  Reacciones:", info.get("reacciones", ""))
                print("  Presentación:", info.get("presentacion", ""))
                print("  Contraindicaciones:", info.get("contraindicaciones", ""))
                print("  Dosis:", info.get("dosis", ""))
                print("  Tipo:", info.get("tipo", ""))
            else:
                print("  (No hay información detallada disponible)")

    print("\n🎯 Proceso finalizado.")
