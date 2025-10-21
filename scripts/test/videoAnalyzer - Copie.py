#!/usr/bin/env python3
"""
face_search_eval.py

But :
- construire une base d'encodages depuis images/persons/
  (chaque image de reference doit contenir principalement le visage et le nom de fichier contient le nom de la personne)
- parcourir images/dataset/train/ pour détecter les visages et reconnaître les personnes
- comparer les predictions au "ground truth" extrait du nom du fichier test (ex: "img_jean_pierre-marie_alice.jpg")
- produire métriques globales et par-personne, et exporter un CSV de résultats

Structure attendue (exemples) :
images/persons/Jean.JPG
images/persons/Alice_1.png
images/dataset/train/img1_Jean_Alice.jpg
images/dataset/train/img2_Bob.jpg
"""

import os
import re
import csv
from collections import defaultdict, Counter

import face_recognition
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Set

# --- Configuration ---
PERSONS_DIR = "../../images/persons"
TEST_DIR = "../../images/dataset train"
OUTPUT_CSV = "results_face_search.csv"

# distance threshold pour considérer une correspondance (plus petit = plus strict)
# valeurs typiques : 0.4 (très strict) - 0.6 (loose). 0.5 est un bon compromis.
TOLERANCE = 0.5

# si tu veux limiter la taille des images lues pour accélérer la détection (préserver ratio)
MAX_IMAGE_DIM = 1600  # mettre None pour pas de redimension

# --- Helpers ---
def normalize_name(s: str) -> str:
    """Nettoie et normalise un nom de fichier en nom lisible (minuscule, pas d'extensions)."""
    # supprime extension
    s = os.path.splitext(os.path.basename(s))[0]
    # remplace séparateurs courants par espace
    s = re.sub(r'[,_\-+;]+', ' ', s)
    # garde les caractères alphabétiques et espace
    s = re.sub(r'[^A-Za-zÀ-ÖØ-öø-ÿ0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s.lower()

def parse_names_from_filename(filename: str, known_names: Set[str]) -> Set[str]:
    """
    Extrait la/les personnes du nom de fichier en comparant les tokens avec known_names.
    On découpe le nom en tokens et on cherche matches pour les noms connus (approx exact).
    """
    base = normalize_name(filename)
    tokens = base.split()
    found = set()
    # essayer correspondances multi-token (ex: "jean pierre")
    for name in known_names:
        norm = name.lower()
        if norm in base:
            found.add(name)
    # fallback : tokens exacts
    for t in tokens:
        for name in known_names:
            if t == name.lower():
                found.add(name)
    return found

def resize_if_needed(img):
    if MAX_IMAGE_DIM is None:
        return img
    h, w = img.shape[:2]
    max_dim = max(h, w)
    if max_dim <= MAX_IMAGE_DIM:
        return img
    scale = MAX_IMAGE_DIM / max_dim
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h))

# --- Build known encodings ---
print("🔎 Construction des encodages connus depuis:", PERSONS_DIR)
known_encodings = []
known_names = []

for fname in os.listdir(PERSONS_DIR):
    path = os.path.join(PERSONS_DIR, fname)
    if not os.path.isfile(path):
        continue
    try:
        image_bgr = cv2.imread(path)
        if image_bgr is None:
            print("⚠️ Impossible de lire :", path)
            continue
        image_bgr = resize_if_needed(image_bgr)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # detecter visages
        face_locations = face_recognition.face_locations(image_rgb, model="hog")
        if len(face_locations) == 0:
            print(f"⚠️ Aucun visage detecté dans reference {fname} -> ignoré.")
            continue

        # on prend le premier visage (supposé être la personne)
        encodings = face_recognition.face_encodings(image_rgb, face_locations)
        if len(encodings) == 0:
            print(f"⚠️ Aucun encoding extrait pour {fname}.")
            continue

        encoding = encodings[0]
        # nominal name from filename (tu peux adapter si tu as "Firstname_Lastname.jpg")
        name = normalize_name(fname)
        # si le fichier contient plusieurs tokens, conserve la forme jointe (ex: "jean pierre" -> "jean pierre")
        # on capitalise commodément :
        name_label = " ".join(name.split())
        name_label = " ".join([w.capitalize() for w in name_label.split()])
        known_encodings.append(encoding)
        known_names.append(name_label)

        print("  ✅ Référence ajoutée :", name_label)

    except Exception as e:
        print("Erreur traitement référence", path, e)

if not known_encodings:
    raise SystemExit("❌ Aucun encodage connu trouvé. Vérifie images/persons/")

known_names_set = set(known_names)
print(f"🔢 {len(known_encodings)} encodages connus chargés.")

# --- Parcours des images test ---
results = []
# metrics accumulators
per_person_stats = {name: {"tp":0, "fp":0, "fn":0} for name in known_names}
global_tp = global_fp = global_fn = 0
n_images = 0

print("\n▶️ Début du parcours des images de test:", TEST_DIR)
for fname in tqdm(sorted(os.listdir(TEST_DIR))):
    path = os.path.join(TEST_DIR, fname)
    if not os.path.isfile(path):
        continue
    n_images += 1
    try:
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            print("⚠️ lecture échouée:", path)
            continue
        img_bgr = resize_if_needed(img_bgr)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # ground truth via nom de fichier
        gt_names = parse_names_from_filename(fname, known_names_set)

        # detecter visages et extraire encodages
        face_locations = face_recognition.face_locations(img_rgb, model="hog")
        encodings = face_recognition.face_encodings(img_rgb, face_locations)

        predicted = set()
        # pour chaque visage, on cherche la meilleure correspondance
        for enc in encodings:
            if len(known_encodings) == 0:
                continue
            distances = face_recognition.face_distance(known_encodings, enc)  # plus petit = plus similaire
            best_idx = np.argmin(distances)
            best_distance = float(distances[best_idx])
            if best_distance <= TOLERANCE:
                predicted_name = known_names[best_idx]
                predicted.add(predicted_name)
            else:
                predicted.add("Unknown")

        # si aucun visage detecté on peut considérer predicted vide (ou Unknown)
        if len(encodings) == 0:
            predicted = set()

        # calcul des TP/FP/FN pour cette image
        # on ignore le label "Unknown" pour les métriques par personne (mais compté en FP global)
        # TP: predicted ∩ gt
        tp_set = set([p for p in predicted if p != "Unknown"]) & set([g.capitalize() for g in gt_names])
        fp_set = set([p for p in predicted if p != "Unknown"]) - set([g.capitalize() for g in gt_names])
        fn_set = set([g.capitalize() for g in gt_names]) - set([p for p in predicted if p != "Unknown"])

        # update global counters
        global_tp += len(tp_set)
        global_fp += len(fp_set) + (1 if "Unknown" in predicted else 0)  # count unknown as fp attempt
        global_fn += len(fn_set)

        # update per-person
        for name in tp_set:
            per_person_stats[name]["tp"] += 1
        for name in fp_set:
            per_person_stats[name]["fp"] += 1
        for name in fn_set:
            per_person_stats[name]["fn"] += 1

        results.append({
            "test_image": fname,
            "ground_truth": ";".join(sorted([g.capitalize() for g in gt_names])) if gt_names else "",
            "predicted": ";".join(sorted(predicted)) if predicted else "",
            "tp": ";".join(sorted(tp_set)) if tp_set else "",
            "fp": ";".join(sorted(fp_set)) if fp_set else "",
            "fn": ";".join(sorted(fn_set)) if fn_set else "",
            "n_faces_detected": len(encodings),
        })

    except Exception as e:
        print("Erreur traitement image test", path, e)

# --- Metrics ---
def safe_div(a,b):
    return a/b if b>0 else 0.0

precision = safe_div(global_tp, global_tp + global_fp)
recall = safe_div(global_tp, global_tp + global_fn)
f1 = safe_div(2*precision*recall, precision+recall) if (precision+recall)>0 else 0.0

print("\n📊 Résultats globaux:")
print(f" Images testées: {n_images}")
print(f" TP: {global_tp}, FP: {global_fp}, FN: {global_fn}")
print(f" Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

# per-person report
per_person_report = []
for name, stats in per_person_stats.items():
    tp = stats["tp"]; fp = stats["fp"]; fn = stats["fn"]
    p = safe_div(tp, tp+fp) if (tp+fp)>0 else None
    r = safe_div(tp, tp+fn) if (tp+fn)>0 else None
    f = safe_div(2*p*r, p+r) if (p and r and (p+r)>0) else None
    per_person_report.append({
        "name": name,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(p,3) if p is not None else None,
        "recall": round(r,3) if r is not None else None,
        "f1": round(f,3) if f is not None else None
    })

# export CSV
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Results exported to {OUTPUT_CSV}")

# print per-person summary
print("\n👥 Résumé par personne:")
for r in per_person_report:
    print(f" - {r['name']}: TP={r['tp']} FP={r['fp']} FN={r['fn']}  P={r['precision']} R={r['recall']} F1={r['f1']}")

# si tu veux charger le CSV dans pandas directement :
# df = pd.read_csv(OUTPUT_CSV)
