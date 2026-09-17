"""Suavizado de landmarks.

El detector de pose entrega coordenadas con temblor de 1-3 px aunque la
persona este quieta. En una proyeccion grande ese temblor se ve como si el
avatar vibrara. El filtro One Euro corta ese ruido cuando hay poco
movimiento y deja pasar el movimiento rapido sin retardo perceptible.
"""

import math

import numpy as np


class OneEuroFilter:
    """One Euro filter vectorizado sobre un arreglo de puntos (N, 2)."""

    def __init__(self, freq=30.0, min_cutoff=1.2, beta=0.02, d_cutoff=1.0):
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = None

    def reset(self):
        self.x_prev = None
        self.dx_prev = None

    def _alpha(self, cutoff):
        # alpha = 1 / (1 + tau/te), con tau = 1/(2*pi*cutoff) y te = 1/freq
        return 1.0 / (1.0 + self.freq / (2.0 * math.pi * cutoff))

    def __call__(self, x, freq=None):
        x = np.asarray(x, dtype=np.float32)

        if freq is not None and freq > 1e-3:
            # Se adapta al FPS real: si la maquina baja de 30 a 15 fps el
            # filtro no se vuelve mas lento de lo esperado.
            self.freq = float(np.clip(freq, 5.0, 120.0))

        if self.x_prev is None or self.x_prev.shape != x.shape:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            return x

        dx = (x - self.x_prev) * self.freq
        a_d = self._alpha(self.d_cutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        # El cutoff sube donde la velocidad es alta -> menos retardo ahi.
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = 1.0 / (1.0 + self.freq / (2.0 * math.pi * cutoff))

        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


class Hysteresis:
    """Evita parpadeo al aparecer/desaparecer la persona frente a la camara."""

    def __init__(self, on_frames=2, off_frames=8):
        self.on_frames = on_frames
        self.off_frames = off_frames
        self._count = 0
        self.state = False

    def update(self, detected):
        if detected == self.state:
            self._count = 0
            return self.state
        self._count += 1
        need = self.on_frames if detected else self.off_frames
        if self._count >= need:
            self.state = detected
            self._count = 0
        return self.state
