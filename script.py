import sqlite3

# Connexion (crée la base si elle n'existe pas)
conn = sqlite3.connect("ma_base.db")

# Création d'un curseur pour exécuter des commandes SQL
cur = conn.cursor()

# Création d'une table "utilisateurs"
cur.execute("""
CREATE TABLE IF NOT EXISTS utilisateurs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nom TEXT NOT NULL,
    age INTEGER
)
""")

# Sauvegarde des changements
conn.commit()

# Fermeture de la connexion
conn.close()

print("✅ Base et table créées avec succès !")
