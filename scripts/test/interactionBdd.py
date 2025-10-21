# =========================================================================================
#TEST INSERTION ET EXTRACTION IMAGE DANS LA BASE

# import sqlite3

# conn = sqlite3.connect("ma_base.db")
# cur = conn.cursor()

# cur.execute("""
# CREATE TABLE IF NOT EXISTS photos (
#     id INTEGER PRIMARY KEY AUTOINCREMENT,
#     nom TEXT NOT NULL,
#     image BLOB NOT NULL
# )
# """)

# print("✅ Table 'photos' créée avec succès !")


# def lire_image(fichier):
#     with open(fichier, 'rb') as f:
#         return f.read()

# conn = sqlite3.connect("ma_base.db")
# cur = conn.cursor()

# nom = "chat"
# image_blob = lire_image("car.png")  # mets ici ton image

# cur.execute("INSERT INTO photos (nom, image) VALUES (?, ?)", (nom, image_blob))

# conn.commit()
# conn.close()

# print("✅ Image insérée avec succès !")

# =========================================================================================

from PIL import Image
import io

conn = sqlite3.connect("ma_base.db")
cur = conn.cursor()

cur.execute("SELECT image FROM photos WHERE nom = ?", ("chat",))
resultat = cur.fetchone()

if resultat:
    image_blob = resultat[0]
    image = Image.open(io.BytesIO(image_blob))
    image.show()  # ouvre l’image avec la visionneuse du système
else:
    print("❌ Image non trouvée.")

conn.close()


import sqlite3
from PIL import Image
import io
import os

# ======================================================
# 1️⃣ Création / connexion à la base
# ======================================================
db_name = "ma_base.db"
conn = sqlite3.connect(db_name)
cur = conn.cursor()
print("✅ Base de données connectée :", db_name)


# ======================================================
# 2️⃣ Création d'une table
# ======================================================
cur.execute("""
CREATE TABLE IF NOT EXISTS photos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nom TEXT NOT NULL,
    image BLOB
)
""")
print("✅ Table 'photos' créée (si elle n’existait pas).")


# ======================================================
# 3️⃣ Insertion de données (exemple avec image)
# ======================================================
def lire_image(fichier):
    with open(fichier, 'rb') as f:
        return f.read()

# Vérifie si le fichier existe avant d'insérer
if os.path.exists("car.png"):
    image_blob = lire_image("car.png")
    cur.execute("INSERT INTO photos (nom, image) VALUES (?, ?)", ("chat", image_blob))
    print("✅ Image insérée dans la table.")
else:
    print("⚠️ Image 'car.png' introuvable, insertion ignorée.")

conn.commit()


# ======================================================
# 4️⃣ Lecture des données
# ======================================================
cur.execute("SELECT id, nom FROM photos")
photos = cur.fetchall()

print("\n📸 Contenu de la table 'photos' :")
for p in photos:
    print(p)


# ======================================================
# 5️⃣ Modification (UPDATE) d’un champ existant
# ======================================================
cur.execute("UPDATE photos SET nom = ? WHERE nom = ?", ("chat_modifié", "chat"))
conn.commit()
print("\n✏️ Nom de l’image modifié avec succès !")


# ======================================================
# 6️⃣ Suppression (DELETE) d’un enregistrement
# ======================================================
cur.execute("DELETE FROM photos WHERE nom = ?", ("chat_modifié",))
conn.commit()
print("🗑️ Enregistrement supprimé avec succès.")


# ======================================================
# 7️⃣ Ajout d’une nouvelle table
# ======================================================
cur.execute("""
CREATE TABLE IF NOT EXISTS utilisateurs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nom TEXT NOT NULL,
    email TEXT UNIQUE
)
""")
conn.commit()
print("\n✅ Nouvelle table 'utilisateurs' ajoutée avec succès.")


# ======================================================
# 8️⃣ Ajout d’une colonne à une table existante
# ======================================================
try:
    cur.execute("ALTER TABLE utilisateurs ADD COLUMN age INTEGER DEFAULT 0")
    conn.commit()
    print("🧱 Colonne 'age' ajoutée à la table 'utilisateurs'.")
except sqlite3.OperationalError:
    print("⚠️ Colonne 'age' existe déjà — aucune modification.")


# ======================================================
# 9️⃣ Insertion et lecture dans la nouvelle table
# ======================================================
cur.execute("INSERT INTO utilisateurs (nom, email, age) VALUES (?, ?, ?)", 
            ("Alice", "alice@example.com", 25))
cur.execute("INSERT INTO utilisateurs (nom, email, age) VALUES (?, ?, ?)", 
            ("Bob", "bob@example.com", 30))
conn.commit()
print("\n✅ Données insérées dans 'utilisateurs'.")

cur.execute("SELECT * FROM utilisateurs")
utilisateurs = cur.fetchall()

print("\n👤 Contenu de la table 'utilisateurs' :")
for u in utilisateurs:
    print(u)


# ======================================================
# 🔟 Suppression d’une table complète
# ======================================================
# ⚠️ À utiliser avec précaution — supprime toute la table !
cur.execute("DROP TABLE IF EXISTS temp_table")
print("\n🧨 Table 'temp_table' supprimée (si elle existait).")


# ======================================================
# 🔁 Exemple : exécuter plusieurs requêtes SQL brutes
# ======================================================
cur.executescript("""
CREATE TABLE IF NOT EXISTS produits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nom TEXT,
    prix REAL
);

INSERT INTO produits (nom, prix) VALUES ('Jus de mangue', 2.5);
INSERT INTO produits (nom, prix) VALUES ('Bissap', 2.0);
""")
conn.commit()
print("\n✅ Requêtes multiples exécutées avec succès.")


# ======================================================
# 1️⃣1️⃣ Lecture avancée : filtrage et tri
# ======================================================
cur.execute("SELECT nom, prix FROM produits WHERE prix > ? ORDER BY prix DESC", (2,))
resultats = cur.fetchall()

print("\n💰 Produits avec prix > 2 :")
for r in resultats:
    print(r)


# ======================================================
# 1️⃣2️⃣ Lecture d’une image depuis la base et affichage
# ======================================================
cur.execute("SELECT image FROM photos LIMIT 1")
resultat = cur.fetchone()

if resultat and resultat[0]:
    image_blob = resultat[0]
    image = Image.open(io.BytesIO(image_blob))
    image.show()
    print("🖼️ Image affichée avec succès.")
else:
    print("❌ Aucune image trouvée dans la table 'photos'.")


# ======================================================
# 1️⃣3️⃣ Fermeture propre de la connexion
# ======================================================
conn.close()
print("\n🔒 Connexion fermée proprement.")

