import cv2

# Charger le classificateur Haar Cascade pour la détection de visages
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialiser la capture vidéo à partir de la webcam
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

    # Convertir la frame en niveaux de gris pour la détection de visages
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détecter les visages dans la frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Dessiner un rectangle autour de chaque visage détecté
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Afficher la frame dans une fenêtre
    cv2.imshow('Flux Webcam avec détection de visages', frame)

    # Arrêter la boucle si l'utilisateur appuie sur 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la capture vidéo et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()
