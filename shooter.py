import cv2
import sys
from ultralytics import YOLO

def ouvrir_webcam(index_camera=0):
    cap = cv2.VideoCapture(index_camera)

    if not cap.isOpened():
        print("Erreur : impossible d'ouvrir la webcam.")
        sys.exit(1)

    return cap


def charger_modele():
    # Modèle léger YOLOv8
    try:
        modele = YOLO("yolov8n.pt")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        sys.exit(1)

    return modele


def encadrer_oiseaux(image, modele, seuil_confiance=0.4):
    resultats = modele(image, verbose=False)

    nb_oiseaux = 0

    for resultat in resultats:
        boites = resultat.boxes

        for boite in boites:
            classe_id = int(boite.cls[0].item())
            confiance = float(boite.conf[0].item())
            nom_classe = modele.names[classe_id]

            # On ne garde que la classe "bird"
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


def afficher_flux_webcam():
    cap = ouvrir_webcam(1)
    modele = charger_modele()

    print("Appuyez sur 'q' pour quitter.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Erreur : image webcam non récupérée.")
            break

        frame, nb_oiseaux = encadrer_oiseaux(frame, modele)

        cv2.putText(
            frame,
            f"Oiseaux detectes : {nb_oiseaux}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("Webcam - Detection d'oiseaux", frame)

        touche = cv2.waitKey(1) & 0xFF
        if touche == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    afficher_flux_webcam()
