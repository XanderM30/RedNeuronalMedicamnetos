import firebase_admin
from firebase_admin import credentials, firestore

# 1️⃣ Ruta a tu archivo JSON de credenciales
SERVICE_ACCOUNT_PATH = "firebase_config/serviceAccountKey.json"  # <- Cambia esto

# 2️⃣ Inicializa la app de Firebase
try:
    cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
    firebase_admin.initialize_app(cred)
    print("✅ Firebase inicializado correctamente")
except Exception as e:
    print("❌ Error al inicializar Firebase:", e)
    exit(1)

# 3️⃣ Conecta al cliente de Firestore
db = firestore.client()

# 4️⃣ Intenta leer la colección de medicamentos
FIREBASE_COLLECTION = "medicamentos"  # <- Cambia al nombre real de tu colección

try:
    docs = db.collection(FIREBASE_COLLECTION).limit(5).get()
    print(f"✅ Se pudieron obtener {len(docs)} documentos de '{FIREBASE_COLLECTION}'")
    for doc in docs:
        print(doc.id, "=>", doc.to_dict())
except Exception as e:
    print("❌ Error al leer la colección:", e)
