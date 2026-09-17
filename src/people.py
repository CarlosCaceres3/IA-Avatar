"""Varias personas a la vez, cada una con su avatar.

MediaPipe puede devolver varias poses por cuadro, pero NO garantiza que
vengan siempre en el mismo orden: la persona que hoy es la primera puede
ser la segunda en el cuadro siguiente. Si se dibujara segun ese orden, los
avatares se intercambiarian entre las personas a cada rato.

Por eso aqui cada persona tiene una "ranura" propia, con su filtro de
suavizado, su tamano de cuerpo y su detector de cara. En cada cuadro las
poses detectadas se emparejan con las ranuras por cercania, asi que quien
ya estaba conserva su avatar aunque cambie el orden.
"""

import numpy as np

from face import FaceTracker
from skeleton import BodyState, L_HIP, L_SHOULDER, R_HIP, R_SHOULDER
from smoothing import OneEuroFilter

# Distancia maxima (en fraccion del ancho del cuadro) para considerar que
# una pose detectada es la misma persona de la ranura. Una persona no se
# teletransporta media pantalla entre dos cuadros consecutivos.
MATCH_DIST = 0.22

# Cuadros que una ranura sobrevive sin ser vista antes de liberarse. Da
# margen a que el detector pierda a alguien un instante sin que pierda su
# avatar ni el tamano ya estabilizado.
KEEP_FRAMES = 12


def _center(landmarks):
    """Centro del cuerpo en coordenadas normalizadas 0..1."""
    idx = (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)
    xs = [landmarks[i].x for i in idx]
    ys = [landmarks[i].y for i in idx]
    return float(np.mean(xs)), float(np.mean(ys))


class Person:
    """Estado propio de una persona: filtros, tamano y cara."""

    def __init__(self, slot, fps, face_model=None):
        self.slot = slot
        self.smoother = OneEuroFilter(freq=fps, min_cutoff=1.1, beta=0.018)
        self.body = BodyState()
        self.face = FaceTracker(face_model) if face_model else None
        self.center = (0.5, 0.5)
        self.landmarks = None
        self.head = None          # (centro, radio) en coordenadas del cuadro
        self.missing = 0
        self.seen = False

    def reset_filters(self):
        """Olvida el suavizado pero conserva la pose de este cuadro.

        Se usa cuando cambia la escala de dibujo: los puntos se mueven de
        golpe y los filtros lo verian como un salto de la persona. Lo que
        NO hay que borrar aqui es la pose, que se va a dibujar enseguida.
        """
        self.smoother.reset()
        self.body.reset()

    def reset(self):
        """Olvida todo: la ranura pasa a ser de otra persona."""
        self.reset_filters()
        if self.face is not None:
            self.face.reset()
        self.landmarks = None
        self.head = None

    @property
    def expression(self):
        return self.face.expression if self.face is not None else None

    def close(self):
        if self.face is not None:
            self.face.close()


class Crowd:
    """Reparte las poses detectadas entre ranuras estables."""

    def __init__(self, max_people, fps, face_model=None):
        self.max_people = max_people
        self.fps = fps
        self.face_model = face_model
        self.people = []

    def _nueva(self):
        if len(self.people) >= self.max_people:
            return None
        persona = Person(len(self.people), self.fps, self.face_model)
        self.people.append(persona)
        return persona

    def update(self, poses):
        """Empareja las poses de este cuadro con las personas ya conocidas.

        Devuelve la lista de personas activas, en orden de ranura para que
        el avatar de cada una no cambie de un cuadro a otro.
        """
        for p in self.people:
            p.seen = False

        libres = [p for p in self.people]
        pendientes = []

        # Primero las que se pueden emparejar con alguien conocido: se
        # recorren por cercania creciente para que dos personas juntas no
        # se roben la ranura entre si.
        parejas = []
        for pose_idx, landmarks in enumerate(poses):
            cx, cy = _center(landmarks)
            for persona in libres:
                d = np.hypot(cx - persona.center[0], cy - persona.center[1])
                if d <= MATCH_DIST:
                    parejas.append((d, pose_idx, persona))
        parejas.sort(key=lambda t: t[0])

        usadas_pose = set()
        usadas_slot = set()
        for _, pose_idx, persona in parejas:
            if pose_idx in usadas_pose or persona.slot in usadas_slot:
                continue
            usadas_pose.add(pose_idx)
            usadas_slot.add(persona.slot)
            self._asignar(persona, poses[pose_idx])

        for pose_idx, landmarks in enumerate(poses):
            if pose_idx not in usadas_pose:
                pendientes.append(landmarks)

        # Las que no coincidieron con nadie: ranura libre o una nueva.
        for landmarks in pendientes:
            persona = next((p for p in self.people if not p.seen), None)
            if persona is None:
                persona = self._nueva()
            if persona is None:
                break                      # ya se alcanzo el maximo
            persona.reset()                # es alguien distinto al de antes
            self._asignar(persona, landmarks)

        activas = []
        for p in self.people:
            if p.seen:
                p.missing = 0
                activas.append(p)
            else:
                p.missing += 1
                if p.missing == KEEP_FRAMES:
                    p.reset()
                p.landmarks = None
        return activas

    def _asignar(self, persona, landmarks):
        persona.landmarks = landmarks
        persona.center = _center(landmarks)
        persona.seen = True

    def close(self):
        for p in self.people:
            p.close()
        self.people = []
