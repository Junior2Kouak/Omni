import os
import re
import pandas as pd
from deepface import DeepFace
from tqdm import tqdm

# --- Configuration ---
PERSONS_DIR = "../../images/persons"
TEST_DIR = "../../images/dataset train"
OUTPUT_CSV = "results_deepface.csv"
MODEL_NAME = "VGG-Face"  # alternatives : "Facenet", "ArcFace", "DeepFace"
DETECTOR_BACKEND = "opencv"  # plus précis que default 'opencv', alternatives : mtcnn, retinaface
ENFORCE_DETECTION = True      # True pour éviter les faux positifs
DISTANCE_METRIC = "cosine"    # distance pour DeepFace.find
TOLERANCE = 0.4               # seuil de similarité (plus petit = plus strict)

# --- Helpers ---
def normalize_name(s: str) -> str:
    s = os.path.splitext(os.path.basename(s))[0]
    s = re.sub(r'[,_\-+;]+', ' ', s)
    s = re.sub(r'[^A-Za-zÀ-ÖØ-öø-ÿ0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s.lower()

def parse_names_from_filename(filename: str, known_names: set) -> set:
    base = normalize_name(filename)
    found = set()
    for name in known_names:
        if name.lower() in base:
            found.add(name)
    return found

# --- Préparer les noms connus ---
known_names = []
for f in os.listdir(PERSONS_DIR):
    if f.lower().endswith((".jpg", ".png", ".jpeg")):
        name = normalize_name(f)
        name = " ".join([w.capitalize() for w in name.split()])
        known_names.append(name)
known_names_set = set(known_names)
print(f"🔢 {len(known_names)} personnes de référence chargées : {', '.join(known_names)}")

# --- Parcours des images test ---
results = []

for test_file in tqdm(sorted(os.listdir(TEST_DIR))):
    if not test_file.lower().endswith((".jpg", ".png", ".jpeg")):
        continue
    test_path = os.path.join(TEST_DIR, test_file)
    
    gt_names = parse_names_from_filename(test_file, known_names_set)
    
    try:
        result = DeepFace.find(
            img_path=test_path,
            db_path=PERSONS_DIR,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=ENFORCE_DETECTION,
            distance_metric=DISTANCE_METRIC,
        )

        predicted = set()  # utiliser set pour éviter les doublons
        if len(result) > 0 and not result[0].empty:
            # on filtre par seuil de tolérance
            for i, row in result[0].iterrows():
                distance = row["VGG-Face_cosine"] if DISTANCE_METRIC=="cosine" else row["VGG-Face_euclidean"]
                if distance <= TOLERANCE:
                    name = os.path.splitext(os.path.basename(row["identity"]))[0]
                    name = " ".join([w.capitalize() for w in name.split()])
                    predicted.add(name)

        # TP / FP / FN
        tp = predicted & gt_names
        fp = predicted - gt_names
        fn = gt_names - predicted

        results.append({
            "test_image": test_file,
            "ground_truth": ";".join(sorted(gt_names)),
            "predicted": ";".join(sorted(predicted)),
            "tp": ";".join(sorted(tp)),
            "fp": ";".join(sorted(fp)),
            "fn": ";".join(sorted(fn)),
            "n_faces_detected": len(predicted)
        })

    except Exception as e:
        print(f"⚠️ Erreur pour {test_file} : {e}")

# --- Calcul global ---
global_tp = sum(len(r["tp"].split(";")) for r in results if r["tp"])
global_fp = sum(len(r["fp"].split(";")) for r in results if r["fp"])
global_fn = sum(len(r["fn"].split(";")) for r in results if r["fn"])
def safe_div(a,b): return a/b if b>0 else 0.0
precision = safe_div(global_tp, global_tp + global_fp)
recall = safe_div(global_tp, global_tp + global_fn)
f1 = safe_div(2*precision*recall, precision + recall) if (precision+recall)>0 else 0.0

print("\n📊 Résultats globaux :")
print(f" Images testées : {len(results)}")
print(f" TP={global_tp}, FP={global_fp}, FN={global_fn}")
print(f" Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

# --- Export CSV ---
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Résultats sauvegardés dans : {OUTPUT_CSV}")
