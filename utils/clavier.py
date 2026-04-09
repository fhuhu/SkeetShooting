import sys
import threading
import queue
import termios
import tty
import select

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



