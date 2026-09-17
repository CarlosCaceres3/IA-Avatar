"""Salida a camara virtual.

Windows no trae camaras virtuales de fabrica: hace falta un driver que se
registre en el sistema para que Zoom, Meet, OBS o el proyector vean la
salida como si fuera una webcam mas. pyvirtualcam usa el driver que instala
OBS Studio (OBS Virtual Camera).

Si el driver no esta, la app no se cae: sigue mostrando la ventana y avisa
por consola que la camara virtual no esta disponible.
"""


class VirtualCamera:
    """Envoltorio tolerante a fallos sobre pyvirtualcam."""

    def __init__(self, width, height, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.cam = None
        self.device = None
        self.error = None
        self.enabled = False

    def open(self):
        try:
            import pyvirtualcam
            from pyvirtualcam import PixelFormat
        except ImportError as exc:
            self.error = "falta el paquete pyvirtualcam (" + str(exc) + ")"
            return False

        try:
            # BGR evita convertir el color en cada cuadro: OpenCV ya trabaja asi.
            self.cam = pyvirtualcam.Camera(width=self.width, height=self.height,
                                           fps=self.fps, fmt=PixelFormat.BGR,
                                           print_fps=False)
            self.device = self.cam.device
            self.enabled = True
            return True
        except Exception as exc:
            self.error = str(exc)
            self.cam = None
            return False

    def send(self, frame_bgr):
        if not (self.enabled and self.cam):
            return
        try:
            self.cam.send(frame_bgr)
            self.cam.sleep_until_next_frame()
        except Exception as exc:
            self.error = str(exc)
            self.enabled = False

    def toggle(self):
        if self.enabled:
            self.enabled = False
        elif self.cam is not None:
            self.enabled = True
        else:
            self.open()
        return self.enabled

    def status(self):
        if self.enabled and self.device:
            return "VirtualCam: " + str(self.device)
        if self.cam is not None:
            return "VirtualCam: en pausa (V)"
        return "VirtualCam: no disponible"

    def close(self):
        if self.cam is not None:
            try:
                self.cam.close()
            except Exception:
                pass
        self.cam = None
        self.enabled = False
