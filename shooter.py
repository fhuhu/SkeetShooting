import cv2
import sys
import os
import time
import argparse
from datetime import datetime
from ultralytics import YOLO
import sounddevice as sd
import soundfile as sf
import threading


def ouvrir_webcam(index_camera=0):
    cap = cv2.VideoCapture(index_camera)

    if not cap.isOpened():
        print("Erreur : impossible d'ouvrir la webcam.")
        sys.exit(1)

    return cap


def charger_modele():
    try:
        modele = YOLO("yolov8n.pt")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        sys.exit(1)

    return modele


def creer_dossier_captures(dossier="captures"):
    os.makedirs(dossier, exist_ok=True)
    return dossier


def capturer_image(image, dossier="captures"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nom_fichier = os.path.join(dossier, f"oiseau_{timestamp}.jpg")
    cv2.imwrite(nom_fichier, image)
    print(f"Capture enregistrée : {nom_fichier}")


def jouer_son(fichier="alert.wav"):
    def _play():
        try:
            data, samplerate = sf.read(fichier)
            print(f"Lecture de {fichier} (fs={samplerate} Hz)")
            sd.play(data, samplerate)
            sd.wait()
        except Exception as e:
            print(f"Erreur : {e}")

    threading.Thread(target=_play, daemon=True).start()


def encadrer_oiseaux(image, modele, seuil_confiance=0.4):
    resultats = modele(image, verbose=False)
    nb_oiseaux = 0

    for resultat in resultats:
        for boite in resultat.boxes:
            classe_id = int(boite.cls[0].item())
            confiance = float(boite.conf[0].item())
            nom_classe = modele.names[classe_id]

            if nom_classe == "bird" and confiance >= seuil_confiance:
                x1, y1, x2, y2 = map(int, boite.xyxy[0].tolist())

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    image,
                    f"Oiseau {confiance:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                nb_oiseaux += 1

    return image, nb_oiseaux


def afficher_flux_webcam(visible=False):
    cap = ouvrir_webcam(0)
    modele = charger_modele()
    dossier_captures = creer_dossier_captures("captures")

    if visible:
        print("Appuyez sur 'q' pour quitter.")

    alerte_active = False
    dernier_enregistrement = 0
    delai_capture = 5

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Erreur : image webcam non récupérée.")
            break

        frame_annotee = frame.copy()
        frame_annotee, nb_oiseaux = encadrer_oiseaux(frame_annotee, modele)

        temps_actuel = time.time()

        if nb_oiseaux > 0:
            if not alerte_active:
                jouer_son("alert.wav")
                alerte_active = True

            if temps_actuel - dernier_enregistrement >= delai_capture:
                capturer_image(frame_annotee, dossier_captures)
                dernier_enregistrement = temps_actuel
        else:
            alerte_active = False

        if visible:
            cv2.putText(
                frame_annotee,
                f"Oiseaux detectes : {nb_oiseaux}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.imshow("Webcam - Detection d'oiseaux", frame_annotee)

            touche = cv2.waitKey(1) & 0xFF
            if touche == ord('q'):
                break

    cap.release()

    if visible:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--visible",
        action="store_true",
        help="Afficher la fenêtre graphique"
    )
    args = parser.parse_args()

    afficher_flux_webcam(visible=args.visible)
