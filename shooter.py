import cv2
import sys


def ouvrir_webcam(index_camera=0):
    cap = cv2.VideoCapture(index_camera)

    if not cap.isOpened():
        print("Erreur : impossible d'ouvrir la webcam.")
        sys.exit(1)

    return cap


def charger_detecteur_visage():
    chemin_cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detecteur = cv2.CascadeClassifier(chemin_cascade)

    if detecteur.empty():
        print("Erreur : impossible de charger le détecteur de visages.")
        sys.exit(1)

    return detecteur


def encadrer_visages(image, detecteur):
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    visages = detecteur.detectMultiScale(
        gris,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    for (x, y, w, h) in visages:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image, visages


def afficher_flux_webcam():
    cap = ouvrir_webcam(1)
    detecteur = charger_detecteur_visage()

    print("Appuyez sur 'q' pour quitter.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Erreur : image webcam non récupérée.")
            break

        frame, visages = encadrer_visages(frame, detecteur)

        cv2.putText(
            frame,
            f"Visages detectes : {len(visages)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("Webcam - Detection de visage", frame)

        touche = cv2.waitKey(1) & 0xFF
        if touche == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    afficher_flux_webcam()
