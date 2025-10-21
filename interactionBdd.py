import sqlite3

conn = sqlite3.connect("ma_base.db")
cur = conn.cursor()

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
