#!/usr/bin/env python3
"""
scheduled_record.py

Usage example:
python scheduled_record.py --start "2025-10-21 22:00:00" --end "2025-10-21 22:00:30" --output "out.mp4"
"""

import cv2
import time
from datetime import datetime, timedelta
import argparse
import sys

def parse_datetime(s: str) -> datetime:
    """Parse une string 'YYYY-MM-DD HH:MM:SS' ou 'HH:MM:SS' (aujourd'hui)."""
    try:
        # essayer format complet
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        # sinon essayer seulement l'heure (aujourd'hui)
        try:
            t = datetime.strptime(s, "%H:%M:%S").time()
            today = datetime.now().date()
            return datetime.combine(today, t)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "Format invalide. Utilise 'YYYY-MM-DD HH:MM:SS' ou 'HH:MM:SS'"
            )

def wait_until(target: datetime):
    """Attend jusqu'à l'instant target (affiche compte à rebours simple)."""
    while True:
        now = datetime.now()
        if now >= target:
            return
        remaining = (target - now).total_seconds()
        # sleep par petits pas pour pouvoir réagir au Ctrl+C
        time.sleep(min(1.0, remaining))

def record_between(start_time: datetime, end_time: datetime, output_file: str, camera_index=0, fps=20.0):
    """Enregistre de start_time jusqu'à end_time sur output_file"""
    # si end <= start -> on considère le lendemain
    if end_time <= start_time:
        print("⚠️ end_time <= start_time -> on enregistre jusqu'au lendemain (ajout d'un jour à end_time).")
        end_time += timedelta(days=1)

    now = datetime.now()
    if start_time > now:
        print(f"⏳ En attente du début : {start_time} (maintenant : {now})")
        wait_until(start_time)
    else:
        print(f"▶️ Démarrage immédiat (start_time {start_time} est dans le passé)")

    # Ouvrir la caméra
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("❌ Impossible d'ouvrir la caméra.")
        return

    # Déterminer la résolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    print(f"📷 Résolution : {width}x{height}, fps (caméra) : {actual_fps}")

    # FourCC & writer (MP4)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # ou 'XVID'
    out = cv2.VideoWriter(output_file, fourcc, actual_fps, (width, height))

    print(f"🔴 Enregistrement démarré : {datetime.now()} -> fichier : {output_file}")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Frame non lue (fin probable de la caméra).")
                break

            out.write(frame)

            # Optionnel : afficher la fenêtre (décommenter si besoin)
            # cv2.imshow('Recording', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     print("⏹️ Arrêt manuel par l'utilisateur (touche q)")
            #     break

            # Arrêter si on dépasse l'heure de fin
            if datetime.now() >= end_time:
                print(f"⏹️ Heure de fin atteinte : {datetime.now()}")
                break

            # limiter la boucle pour ne pas surcharger CPU (déjà limité par lecture caméra)
            # small sleep could help if camera is faster than needed
            # time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n🛑 Enregistrement interrompu par l'utilisateur (KeyboardInterrupt).")
    finally:
        cap.release()
        out.release()
        # cv2.destroyAllWindows()
        print(f"✅ Enregistrement terminé, fichier sauvegardé : {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Enregistrer la webcam entre deux heures spécifiées.")
    parser.add_argument("--start", required=True, type=parse_datetime,
                        help="Heure de début: 'YYYY-MM-DD HH:MM:SS' ou 'HH:MM:SS' (aujourd'hui)")
    parser.add_argument("--end", required=True, type=parse_datetime,
                        help="Heure de fin: 'YYYY-MM-DD HH:MM:SS' ou 'HH:MM:SS' (aujourd'hui). Si end <= start, on l'interprete comme le lendemain.")
    parser.add_argument("--output", default="record.mp4", help="Nom de fichier de sortie (mp4).")
    parser.add_argument("--camera", type=int, default=0, help="Index de la caméra (défaut 0).")
    parser.add_argument("--fps", type=float, default=20.0, help="FPS souhaité (utilisé si la caméra ne renvoie pas le fps).")
    args = parser.parse_args()

    record_between(args.start, args.end, args.output, camera_index=args.camera, fps=args.fps)

if __name__ == "__main__":
    main()
