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

# Spécifique Unix/Linux pour lire le clavier sans blocage
import termios
import tty
import select


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


class ClavierTerminal:
    """
    Lecture non bloquante du clavier dans un terminal Unix/Linux.
    Touche 'v' : toggle affichage
    Touche 'q' : quitter
    """
    def __init__(self):
        self.file = sys.stdin
        self.fd = self.file.fileno()
        self.old_settings = None
        self.touches = queue.Queue()
        self.actif = False
        self.thread = None

    def demarrer(self):
        if not self.file.isatty():
            print("Attention : stdin n'est pas un terminal. Les touches v/q ne seront pas capturées.")
            return

        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        self.actif = True
        self.thread = threading.Thread(target=self._boucle_lecture, daemon=True)
        self.thread.start()

    def _boucle_lecture(self):
        while self.actif:
            try:
                rlist, _, _ = select.select([self.file], [], [], 0.1)
                if rlist:
                    caractere = self.file.read(1)
                    if caractere:
                        self.touches.put(caractere)
            except Exception:
                break

    def lire_touches(self):
        touches = []
        while not self.touches.empty():
            try:
                touches.append(self.touches.get_nowait())
            except queue.Empty:
                break
        return touches

    def arreter(self):
        self.actif = False
        if self.old_settings is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)


def afficher_flux_webcam(visible=False):
    cap = ouvrir_webcam(1)
    modele = charger_modele()
    dossier_captures = creer_dossier_captures("captures")
    clavier = ClavierTerminal()

    print("Commandes clavier :")
    print("  v : afficher/masquer la fenêtre")
    print("  q : quitter")
    print("Le terminal doit avoir le focus pour capter les touches.")

    alerte_active = False
    dernier_enregistrement = 0
    delai_capture = 5  # secondes entre deux captures
    fenetre_ouverte = False

    try:
        clavier.demarrer()

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Erreur : image webcam non récupérée.")
                break

            frame_annotee = frame.copy()
            frame_annotee, nb_oiseaux = encadrer_oiseaux(frame_annotee, modele)

            temps_actuel = time.time()

            # Déclenchement du son et captures
            if nb_oiseaux > 0:
                if not alerte_active:
                    jouer_son("alert.wav")
                    alerte_active = True

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

                cv2.imshow("Webcam - Detection d'oiseaux", frame_annotee)
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
        "-v", "--visible",
        action="store_true",
        help="Afficher la fenêtre graphique au démarrage"
    )
    args = parser.parse_args()

    afficher_flux_webcam(visible=args.visible)

