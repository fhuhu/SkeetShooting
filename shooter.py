import cv2

# Initialiser la capture vidéo à partir de la webcam (0 est généralement l'ID de la webcam par défaut)
cap = cv2.VideoCapture(0)

# Vérifier si la webcam est ouverte correctement
if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la webcam.")
    exit()

# Boucle pour lire et afficher chaque frame de la webcam
while True:
    ret, frame = cap.read()

    # Si la frame n'est pas lue correctement, sortir de la boucle
    if not ret:
        print("Erreur : Impossible de lire la frame.")
        break

    # Afficher la frame dans une fenêtre
    cv2.imshow('Flux Webcam', frame)

    # Arrêter la boucle si l'utilisateur appuie sur 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la capture vidéo et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()
