import cv2
import sys
import os
import time
import argparse
import threading
import queue
from datetime import datetime
from ultralytics import YOLO
import sounddevice as sd
import soundfile as sf
import yaml
from utils.clavier import ClavierTerminal
from utils.default_config import DEFAULT_CONFIG

def fusion_dicts(base, override):
    resultat = dict(base)
    for cle, valeur in override.items():
        if cle in resultat and isinstance(resultat[cle], dict) and isinstance(valeur, dict):
            resultat[cle] = fusion_dicts(resultat[cle], valeur)
        else:
            resultat[cle] = valeur

    return resultat

def charger_config(chemin="config.yaml"):
    config = DEFAULT_CONFIG
    if os.path.exists(chemin):
        with open(chemin, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}
        config = fusion_dicts(DEFAULT_CONFIG, user_config)
        print(f"Configuration chargée depuis : {chemin}")
    else:
        print(f"Fichier {chemin} introuvable, utilisation de la configuration par défaut.")

    return config

def generer_config_si_absent(chemin="config.yaml"):
    if not os.path.exists(chemin):
        import yaml
        with open(chemin, "w", encoding="utf-8") as f:
            yaml.dump(DEFAULT_CONFIG, f, sort_keys=False)
        print(f"Fichier de config créé : {chemin}")


def ouvrir_webcam(index_camera=0):
    cap = cv2.VideoCapture(index_camera)

    if not cap.isOpened():
        print("Erreur : impossible d'ouvrir la webcam.")
        sys.exit(1)

    return cap


def charger_modele(model):
    try:
        modele = YOLO(model)
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
    """
    Joue un son sans bloquer le programme (thread séparé)
    """
    def _play():
        try:
            data, samplerate = sf.read(fichier)
            print(f"Lecture de {fichier} (fs={samplerate} Hz)")
            sd.play(data, samplerate)
            sd.wait()
        except Exception as e:
            print(f"Erreur audio : {e}")

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

                # Centre de la boîte englobante
                x_centre = (x1 + x2) // 2
                y_centre = (y1 + y2) // 2

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(image, (x_centre, y_centre), 4, (0, 0, 255), -1)

                cv2.putText(
                    image,
                    f"Trust {confiance:.2f} (x={x_centre}, y={y_centre})",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                nb_oiseaux += 1

    return image, nb_oiseaux


def afficher_flux_webcam(config, visible=False):
    cap = ouvrir_webcam(config["camera"]["index"])
    modele = charger_modele(config["model"]["weights"])
    nom_fenetre = config["display"]["window_name"]
    dossier_captures = creer_dossier_captures(config["captures"]["directory"])
    clavier = ClavierTerminal()

    print("Commandes clavier :")
    print("  v : afficher/masquer la fenêtre")
    print("  q : quitter")
    print("Le terminal doit avoir le focus pour capter les touches.")

    alerte_active = False
    dernier_enregistrement = 0
    delai_capture = config["detection"]["capture_delay_seconds"]  # secondes entre deux captures
    fenetre_ouverte = False

    try:
        clavier.demarrer()

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Erreur : image webcam non récupérée.")
                break

            frame_annotee = frame.copy()
            # Centre de l'image
            h, w = frame_annotee.shape[:2]
            centre_x = w // 2
            centre_y = h // 2

            # Point rouge au centre
            cv2.circle(frame_annotee, (centre_x, centre_y), 5, (0, 0, 255), -1)

            frame_annotee, nb_oiseaux = encadrer_oiseaux(frame_annotee, modele)

            temps_actuel = time.time()

            # Déclenchement du son et captures
            if nb_oiseaux > 0:
                if temps_actuel - dernier_enregistrement >= delai_capture:
                    capturer_image(frame_annotee, dossier_captures)
                    dernier_enregistrement = temps_actuel
            else:
                alerte_active = False

            # Gestion des touches clavier
            for touche in clavier.lire_touches():
                if touche.lower() == "v":
                    visible = not visible
                    print(f"Affichage {'activé' if visible else 'désactivé'}")

                    if not visible and fenetre_ouverte:
                        cv2.destroyAllWindows()
                        fenetre_ouverte = False

                elif touche.lower() == "q":
                    print("Arrêt du programme.")
                    return

            # Affichage conditionnel
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

                cv2.imshow(config["display"]["window_name"], frame_annotee)
                fenetre_ouverte = True

                # Nécessaire pour rafraîchir la fenêtre OpenCV
                # et permet aussi de fermer via le gestionnaire de fenêtre
                touche_fenetre = cv2.waitKey(1) & 0xFF
                if touche_fenetre == ord('q'):
                    print("Arrêt du programme.")
                    return
                elif touche_fenetre == ord('v'):
                    visible = False
                    cv2.destroyAllWindows()
                    fenetre_ouverte = False
                    print("Affichage désactivé")
            else:
                if fenetre_ouverte:
                    cv2.destroyAllWindows()
                    fenetre_ouverte = False

                # petite pause pour éviter de saturer le CPU
                time.sleep(0.01)

    finally:
        clavier.arreter()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c", "--config",
        default="config.yaml",
        help="Chemin du fichier de configuration YAML"
    )

    parser.add_argument(
        "-v", "--visible",
        action="store_true",
        help="Afficher la fenêtre graphique au démarrage"
    )

    args = parser.parse_args()

    generer_config_si_absent(chemin=args.config)

    config = charger_config(args.config)

    # Si -v est fourni, il force l'affichage à True
    visible_override = True if args.visible else None

    afficher_flux_webcam(config, visible=visible_override)

