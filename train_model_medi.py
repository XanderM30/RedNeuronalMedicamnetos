"""
entrenar_medicamentos_incremental.py
Pipeline completo para:
- Descargar colección 'medicamentos' (y subcolecciones)
- Entrenar/reentrenar un modelo de clasificación por 'nombre'
- Guardar artefactos (.h5, .tflite, vocab.json, label_map.json, med_documents.json, model_report.json)
- Versionar en version.txt (MAJOR.MINOR.PATCH -> incrementa PATCH automáticamente)
- Detectar cambios en Firebase y reentrenar si hay nuevos documentos
- Inferencia con Keras o TFLite

Requisitos:
pip install firebase-admin tensorflow scikit-learn pandas numpy
"""

import os
import json
import shutil
from datetime import datetime
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import firebase_admin
from firebase_admin import credentials, firestore

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ---------------------------
# CONFIGURACIÓN (ajusta aquí)
# ---------------------------
SERVICE_ACCOUNT_PATH = "firebase_config/serviceAccountKey.json"  # <- tu key
FIREBASE_COLLECTION = "medicamentos"
OUTPUT_DIR = "model_output"
BACKUP_DIR = os.path.join(OUTPUT_DIR, "backup")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)

# Hyperparámetros
BATCH_SIZE = 32
EPOCHS = 30
VALIDATION_SPLIT = 0.15
MAX_VOCAB_SIZE = 5000
SEQUENCE_LENGTH = 32
EMBEDDING_DIM = 64
RANDOM_SEED = 42

VERSION_FILE = os.path.join(OUTPUT_DIR, "version.txt")
MODEL_H5 = os.path.join(OUTPUT_DIR, "med_classifier.h5")
TFLITE_PATH = os.path.join(OUTPUT_DIR, "med_classifier.tflite")
VOCAB_PATH = os.path.join(OUTPUT_DIR, "vocab.json")
LABEL_MAP_PATH = os.path.join(OUTPUT_DIR, "label_map.json")
DOCS_PATH = os.path.join(OUTPUT_DIR, "med_documents.json")
MODEL_REPORT_PATH = os.path.join(OUTPUT_DIR, "model_report.json")

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---------------------------
# Utilidades Firebase
# ---------------------------
def init_firestore(sa_path):
    if not firebase_admin._apps:
        cred = credentials.Certificate(sa_path)
        firebase_admin.initialize_app(cred)
    return firestore.client()

def fetch_all_medications(db, collection_name):
    meds = []
    col_ref = db.collection(collection_name)
    docs = col_ref.stream()
    for d in docs:
        data = d.to_dict() or {}
        data['_id'] = d.id
        # buscar subcolecciones
        subcols = {}
        for subcol in d.reference.collections():
            items = []
            for sd in subcol.stream():
                sdat = sd.to_dict() or {}
                sdat['_id'] = sd.id
                items.append(sdat)
            subcols[subcol.id] = items
        if subcols:
            data['subcollections'] = subcols
        meds.append(data)
    return meds

# ---------------------------
# Prepare dataset
# ---------------------------
def prepare_dataset(med_list):
    rows = []
    id_to_doc = {}
    for m in med_list:
        nombre = (m.get('nombre') or "").strip()
        if not nombre:
            continue
        docid = m['_id']
        rows.append({"nombre": nombre, "label": docid})
        id_to_doc[docid] = m
    df = pd.DataFrame(rows)
    return df, id_to_doc

# ---------------------------
# TextVectorization & Model
# ---------------------------
def build_text_vectorizer(texts, max_tokens=MAX_VOCAB_SIZE, seq_len=SEQUENCE_LENGTH):
    vect = layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode='int',
        output_sequence_length=seq_len
    )
    vect.adapt(texts)
    return vect

def build_model(vocab_size, seq_len, n_classes, embedding_dim=EMBEDDING_DIM):
    # modelo que recibe ints de longitud seq_len
    inputs = layers.Input(shape=(seq_len,), dtype='int32', name='input_ids')
    x = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=seq_len)(inputs)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(n_classes, activation='softmax', name='output')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# ---------------------------
# Versioning
# ---------------------------
VERSION_FILE = os.path.join(OUTPUT_DIR, "version.txt")

def read_version():
    """Lee un número entero desde version.txt; si no existe, devuelve 0"""
    if not os.path.exists(VERSION_FILE):
        return 0
    try:
        return int(open(VERSION_FILE, "r").read().strip())
    except Exception:
        return 0

def bump_version():
    """Incrementa la versión en 1 y la guarda"""
    version = read_version() + 1
    with open(VERSION_FILE, "w") as f:
        f.write(str(version))
    return version


# ---------------------------
# Guardar backup
# ---------------------------
def backup_old_artifacts():
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    dest = os.path.join(BACKUP_DIR, f"backup_{ts}")
    os.makedirs(dest, exist_ok=True)
    for p in [MODEL_H5, TFLITE_PATH, VOCAB_PATH, LABEL_MAP_PATH, DOCS_PATH, MODEL_REPORT_PATH]:
        if os.path.exists(p):
            shutil.copy(p, dest)

# ---------------------------
# Train pipeline
# ---------------------------
def train_and_export(sa_path, quantize=False, quantize_int=False, representative_texts=None):
    db = init_firestore(sa_path)
    med_list = fetch_all_medications(db, FIREBASE_COLLECTION)
    print(f"[INFO] Documentos descargados: {len(med_list)}")
    df, id_to_doc = prepare_dataset(med_list)
    n_samples = len(df)
    if n_samples == 0:
        raise RuntimeError("No hay medicamentos con campo 'nombre' en la colección.")

    # label encode
    le = LabelEncoder()
    y = le.fit_transform(df['label'])
    labels = le.classes_.tolist()  # estos son los document ids en orden de índice

    # vectorizer
    vectorizer = build_text_vectorizer(df['nombre'].values, max_tokens=MAX_VOCAB_SIZE, seq_len=SEQUENCE_LENGTH)
    vocab = vectorizer.get_vocabulary()
    vocab_size = len(vocab)
    print(f"[INFO] Vocab size: {vocab_size}")

    # secuencias ints
    # TextVectorization espera batches de strings shaped (N,) or (N,1) if using adapt; we'll pass plain array
    X_int = vectorizer(np.array(df['nombre'].values)).numpy()

    # construir modelo
    n_classes = len(labels)
    model = build_model(vocab_size, SEQUENCE_LENGTH, n_classes, embedding_dim=EMBEDDING_DIM)
    model.summary()

    # callbacks
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint(MODEL_H5, monitor='val_loss', save_best_only=True, save_weights_only=False)
]


    # entrenar
    history = model.fit(
        X_int,
        y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=2
    )

    # métricas finales
    train_loss = float(history.history['loss'][-1])
    train_acc = float(history.history['accuracy'][-1])
    val_loss = float(history.history['val_loss'][-1]) if 'val_loss' in history.history else None
    val_acc = float(history.history['val_accuracy'][-1]) if 'val_accuracy' in history.history else None

    # evaluación en todo dataset
    probs = model.predict(X_int)
    preds = np.argmax(probs, axis=1)
    cls_report = classification_report(y, preds, output_dict=True, zero_division=0)
    confmat = confusion_matrix(y, preds).tolist()

    # métricas resumen
    overall_accuracy = cls_report.get("accuracy", None)
    error_percent = None
    if overall_accuracy is not None:
        error_percent = (1.0 - overall_accuracy) * 100.0

    # versionado
    # backup antes de sobreescribir
    backup_old_artifacts()
    version = bump_version()
    print(f"[INFO] Versión del modelo: {version}")
     

    # guardar artefactos
    # 1) vocab
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    # 2) label_map (index -> doc id)
    label_map = {"classes": labels}
    with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    # 3) documentos originales
    with open(DOCS_PATH, "w", encoding="utf-8") as f:
        json.dump(id_to_doc, f, ensure_ascii=False, indent=2)

    # 4) guardar modelo Keras
    model.save(MODEL_H5)

    # 5) reporte
    model_report = {
        "version": version,
        "timestamp": ts,
        "model_type": "Clasificador (Embedding + GAP + Dense)",
        "tensorflow_version": tf.__version__,
        "hyperparameters": {
            "batch_size": BATCH_SIZE,
            "epochs_requested": EPOCHS,
            "epochs_run": len(history.history['loss']),
            "validation_split": VALIDATION_SPLIT,
            "max_vocab_size": MAX_VOCAB_SIZE,
            "sequence_length": SEQUENCE_LENGTH,
            "embedding_dim": EMBEDDING_DIM,
            "random_seed": RANDOM_SEED
        },
        "dataset": {
            "n_samples": int(n_samples),
            "n_classes": int(n_classes)
        },
        "training_metrics": {
            "train_loss_final": train_loss,
            "train_accuracy_final": train_acc,
            "val_loss_final": val_loss,
            "val_accuracy_final": val_acc,
            "overall_accuracy": overall_accuracy,
            "error_percent": error_percent
        },
        "classification_report": cls_report,
        "confusion_matrix": confmat,
        "notes": "Cada documento de Firebase se codifica como una clase (label = doc id)."
    }
    with open(MODEL_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(model_report, f, ensure_ascii=False, indent=2)

    print("[INFO] Artefactos guardados en:", OUTPUT_DIR)

    # 6) Convertir a TFLite (sin cuantización por defecto)
    try:
        convert_to_tflite(MODEL_H5, VOCAB_PATH, TFLITE_PATH, quantize=quantize, quantize_int=quantize_int, representative_texts=(representative_texts or df['nombre'].tolist()))
        print("[INFO] Conversión TFLite completada.")
    except Exception as e:
        print("[WARN] Falló la conversión a TFLite:", e)

    return {
        "model": model,
        "vectorizer_vocab": vocab,
        "label_map": label_map,
        "id_to_doc": id_to_doc,
        "report": model_report,
        "paths": {
            "model_h5": MODEL_H5,
            "tflite": TFLITE_PATH,
            "vocab": VOCAB_PATH,
            "label_map": LABEL_MAP_PATH,
            "docs": DOCS_PATH,
            "report": MODEL_REPORT_PATH,
            "version_file": VERSION_FILE
        }
    }

# ---------------------------
# Convertir a TFLite
# ---------------------------
def convert_to_tflite(keras_model_path, vectorizer_vocab_path, tflite_path, quantize=False, quantize_int=False, representative_texts=None):
    """
    crea un modelo completo que recibe string -> TextVectorization -> modelo_keras (ints)
    y lo convierte a TFLite. Si quantize=True se hará post-training quantization (float16 or int8)
    Si quantize_int=True -> full integer quantization (requiere representative_texts)
    """
    with open(vectorizer_vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    seq_len = SEQUENCE_LENGTH
    vect = layers.TextVectorization(max_tokens=len(vocab), output_mode='int', output_sequence_length=seq_len)
    vect.set_vocabulary(vocab)

    keras_model = tf.keras.models.load_model(keras_model_path)

    # crear modelo completo
    text_input = tf.keras.Input(shape=(1,), dtype=tf.string, name='input_text')
    x = vect(text_input)
    outputs = keras_model(x)  # el keras_model espera ints
    full_model = tf.keras.Model(text_input, outputs)

    # convertir
    converter = tf.lite.TFLiteConverter.from_keras_model(full_model)
    converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,     # Operaciones estándar
    tf.lite.OpsSet.SELECT_TF_OPS        # Permite TextVectorization y similares
]

    if quantize_int:
        # Full integer quantization
        if not representative_texts:
            raise ValueError("representative_texts required for full integer quantization.")
        def representative_gen():
            for t in representative_texts[:100]:
                # devolver batch de strings
                yield [np.array([t], dtype=object)]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    elif quantize:
        # float16 quantization (reducción de tamaño conservadora)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

# ---------------------------
# Inferencia (Keras)
# ---------------------------
def load_artifacts_for_inference():
    if not os.path.exists(MODEL_H5):
        raise FileNotFoundError("No se encontró modelo Keras en " + MODEL_H5)
    if not os.path.exists(VOCAB_PATH):
        raise FileNotFoundError("No se encontró vocab.json en " + VOCAB_PATH)
    if not os.path.exists(LABEL_MAP_PATH):
        raise FileNotFoundError("No se encontró label_map.json en " + LABEL_MAP_PATH)
    if not os.path.exists(DOCS_PATH):
        raise FileNotFoundError("No se encontró med_documents.json en " + DOCS_PATH)

    model = tf.keras.models.load_model(MODEL_H5)
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    vect = layers.TextVectorization(max_tokens=len(vocab), output_mode='int', output_sequence_length=SEQUENCE_LENGTH)
    vect.set_vocabulary(vocab)
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    with open(DOCS_PATH, "r", encoding="utf-8") as f:
        id_to_doc = json.load(f)
    return model, vect, label_map, id_to_doc

def predict_by_name_keras(name, model, vect, label_map, id_to_doc, top_k=1):
    """
    name: string ingresado por el usuario
    Devuelve: lista de (doc, score) ordenado por score desc
    """
    x_int = vect(np.array([name])).numpy()
    probs = model.predict(x_int)
    probs = probs[0]
    idxs = np.argsort(probs)[::-1][:top_k]
    result = []
    classes = label_map['classes']
    for idx in idxs:
        docid = classes[idx]
        score = float(probs[idx])
        doc = id_to_doc.get(docid, {"_id": docid})
        result.append({"doc": doc, "score": score})
    return result

# ---------------------------
# Inferencia (TFLite)
# ---------------------------
def predict_by_name_tflite(name, tflite_path=TFLITE_PATH, top_k=1):
    if not os.path.exists(tflite_path):
        raise FileNotFoundError("No se encontró TFLite en " + tflite_path)
    if not os.path.exists(VOCAB_PATH) or not os.path.exists(LABEL_MAP_PATH) or not os.path.exists(DOCS_PATH):
        raise FileNotFoundError("Faltan artefactos necesarios (vocab/label_map/docs)")
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    # crear layer vect con vocab local (para convertir texto -> ints)
    vect = layers.TextVectorization(max_tokens=len(vocab), output_mode='int', output_sequence_length=SEQUENCE_LENGTH)
    vect.set_vocabulary(vocab)
    x_int = vect(np.array([name])).numpy()  # shape (1, seq_len)
    # Cargar label_map y docs
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    with open(DOCS_PATH, "r", encoding="utf-8") as f:
        id_to_doc = json.load(f)

    # Inicializar intérprete
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Buscar la entrada que espera STRING o INT; nuestro TFLite completo debería aceptar STRING
    # Si el converted model acepta string, hay que pasarlo como bytes
    # Intentamos primero pasar string bytes; si falla, pasar ints
    try:
        # input espera string
        if input_details[0]['dtype'] == np.dtype('object') or input_details[0]['dtype'] == np.bytes_:

            interpreter.set_tensor(input_details[0]['index'], np.array([name], dtype=object))
        else:
            # suponer que el modelo espera ints (si se convirtió un pipeline diferente)
            interpreter.set_tensor(input_details[0]['index'], x_int.astype(np.int32))
    except Exception:
        # fallback a ints
        interpreter.set_tensor(input_details[0]['index'], np.array([[query.encode('utf-8')]], dtype=np.bytes_))



    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    probs = np.array(output_data[0])
    idxs = np.argsort(probs)[::-1][:top_k]
    result = []
    classes = label_map['classes']
    for idx in idxs:
        docid = classes[int(idx)]
        score = float(probs[int(idx)])
        doc = id_to_doc.get(docid, {"_id": docid})
        result.append({"doc": doc, "score": score})
    return result


# ---------------------------
# Detección de cambios en Firebase
# ---------------------------
def detect_changes_in_firebase(local_docs_path, current_docs):
    """
    Compara los documentos actuales de Firebase con los guardados localmente.
    Si hay diferencias (nuevos, eliminados o modificados), devuelve True.
    """
    if not os.path.exists(local_docs_path):
        print("[⚠️] No existe med_documents.json local. Se entrenará por primera vez.")
        return True

    with open(local_docs_path, "r", encoding="utf-8") as f:
        prev_docs = json.load(f)

    prev_ids = set(prev_docs.keys())
    curr_ids = set([m["_id"] for m in current_docs])

    # documentos nuevos o eliminados
    nuevos = curr_ids - prev_ids
    eliminados = prev_ids - curr_ids

    # detectar cambios en campos (solo compara 'nombre')
    modificados = []
    for mid in curr_ids & prev_ids:
        prev_name = (prev_docs[mid].get("nombre") or "").strip()
        curr_name = ""
        for m in current_docs:
            if m["_id"] == mid:
                curr_name = (m.get("nombre") or "").strip()
                break
        if prev_name != curr_name:
            modificados.append(mid)

    hay_cambios = bool(nuevos or eliminados or modificados)

    print("───────────────────────────────────────────────")
    if hay_cambios:
        print("[🔄] Cambios detectados en Firebase:")
        if nuevos:
            print(f"   🆕 Nuevos documentos: {list(nuevos)}")
        if eliminados:
            print(f"   ❌ Eliminados: {list(eliminados)}")
        if modificados:
            print(f"   ✏️ Modificados: {list(modificados)}")
    else:
        print("[✅] Sin cambios detectados en Firebase. No se requiere reentrenamiento.")
    print("───────────────────────────────────────────────")

    return hay_cambios


# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    print("═══════════════════════════════════════════════════════════")
    print(" 🧠  Iniciando pipeline de entrenamiento de medicamentos")
    print("═══════════════════════════════════════════════════════════")

    db = init_firestore(SERVICE_ACCOUNT_PATH)
    med_list = fetch_all_medications(db, FIREBASE_COLLECTION)
    print(f"[INFO] Documentos descargados desde Firebase: {len(med_list)}")

    # Detectar cambios con respecto al modelo anterior
    cambios = detect_changes_in_firebase(DOCS_PATH, med_list)

    if cambios:
        print("[🚀] Entrenando / Reentrenando modelo con los nuevos datos...")
        out = train_and_export(SERVICE_ACCOUNT_PATH, quantize=False, quantize_int=False)

        print("───────────────────────────────────────────────")
        print(f"[✅] Modelo actualizado correctamente.")
        print(f"[📦] Guardado: {out['paths']['model_h5']}")
        print(f"[🤖] Convertido a TFLite: {out['paths']['tflite']}")
        print(f"[🕓] Versión: {read_version()}")
        print("───────────────────────────────────────────────")

        # Prueba rápida de inferencia
        model, vect, label_map, id_to_doc = load_artifacts_for_inference()
        ejemplo = "Penicilina"
        preds = predict_by_name_keras(ejemplo, model, vect, label_map, id_to_doc, top_k=1)
        print(f"[🔍] Ejemplo de predicción con Keras ({ejemplo}):")
        print(json.dumps(preds, indent=2, ensure_ascii=False))
    else:
        print("[🟢] No se necesita reentrenamiento. Manteniendo modelo anterior.")

    print("═══════════════════════════════════════════════════════════")
    print(" ✅ Proceso finalizado. Modelo listo para Flutter.")
    print("═══════════════════════════════════════════════════════════")
