"""Varias personas a la vez, cada una con su avatar.

MediaPipe puede devolver varias poses por cuadro, pero NO garantiza que
vengan siempre en el mismo orden: la persona que hoy es la primera puede
ser la segunda en el cuadro siguiente. Si se dibujara segun ese orden, los
avatares se intercambiarian entre las personas a cada rato.

Por eso aqui cada persona tiene una "ranura" propia, con su filtro de
suavizado y su tamano de cuerpo. En cada cuadro las poses detectadas se
emparejan con las ranuras, asi que quien ya estaba conserva su avatar
aunque cambie el orden.

El emparejamiento no mira solo la posicion. Cuando dos personas se cruzan
sus centros se juntan y la posicion sola no alcanza para distinguirlas,
asi que ademas se usan:

  - Hacia donde venia moviendose cada una (se predice donde deberia estar).
  - Su altura en el cuadro, que cambia poco entre cuadros y distingue a
    una persona alta de una baja, o a una cerca de una lejos.
"""

import numpy as np

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

# Cuanto pesa la diferencia de altura frente a la distancia. Con 0.35, dos
# personas de altura muy distinta se distinguen aunque sus centros esten
# casi encima, que es justo el caso del cruce.
SIZE_WEIGHT = 0.35

# Cuanto se conserva de la velocidad anterior. Alto, porque interesa la
# tendencia del movimiento y no el temblor cuadro a cuadro.
VEL_SMOOTH = 0.6


def _center(landmarks):
    """Centro del cuerpo en coordenadas normalizadas 0..1."""
    idx = (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)
    xs = [landmarks[i].x for i in idx]
    ys = [landmarks[i].y for i in idx]
    return float(np.mean(xs)), float(np.mean(ys))


def _size(landmarks):
    """Alto aparente de la persona, como fraccion del cuadro.

    Se mide sobre los puntos que se ven con confianza: sirve tanto si la
    persona sale entera como si solo se le ve medio cuerpo. Es la senal
    que distingue a alguien cerca de alguien lejos cuando ambos pasan por
    el mismo sitio de la pantalla.
    """
    ys = [lm.y for lm in landmarks if getattr(lm, "visibility", 1.0) > 0.5]
    if len(ys) < 4:
        return 0.0
    return float(max(ys) - min(ys))


def _cost(persona, cx, cy, size):
    """Cuanto 'cuesta' decir que esa pose es esta persona. Menor es mejor.

    A la distancia se le suma una penalizacion por diferencia de altura.
    Sin eso, dos personas que se cruzan intercambian avatares justo en el
    momento del cruce, que es cuando mas se nota.
    """
    px = persona.center[0] + persona.velocity[0]
    py = persona.center[1] + persona.velocity[1]
    dist = float(np.hypot(cx - px, cy - py))

    if persona.size > 0.01 and size > 0.01:
        rel = abs(size - persona.size) / max(persona.size, size)
        dist += min(rel, 1.0) * SIZE_WEIGHT
    return dist


class Person:
    """Estado propio de una persona: filtros, tamano y movimiento."""

    def __init__(self, slot, fps):
        self.slot = slot
        self.smoother = OneEuroFilter(freq=fps, min_cutoff=1.1, beta=0.018)
        self.body = BodyState()
        self.center = (0.5, 0.5)
        self.velocity = (0.0, 0.0)
        self.size = 0.0           # alto del cuerpo en el cuadro, 0..1
        self.landmarks = None
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
        self.landmarks = None
        self.velocity = (0.0, 0.0)
        self.size = 0.0


class Crowd:
    """Reparte las poses detectadas entre ranuras estables."""

    def __init__(self, max_people, fps):
        self.max_people = max_people
        self.fps = fps
        self.people = []

    def _nueva(self):
        if len(self.people) >= self.max_people:
            return None
        persona = Person(len(self.people), self.fps)
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
        medidas = [(_center(lm)[0], _center(lm)[1], _size(lm)) for lm in poses]

        parejas = []
        for pose_idx, (cx, cy, size) in enumerate(medidas):
            for persona in libres:
                if persona.landmarks is None and persona.missing >= KEEP_FRAMES:
                    continue           # ranura libre: no compite por emparejar
                d = _cost(persona, cx, cy, size)
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
            self._asignar(persona, poses[pose_idx], medidas[pose_idx])

        for pose_idx, landmarks in enumerate(poses):
            if pose_idx not in usadas_pose:
                pendientes.append((landmarks, medidas[pose_idx]))

        # Las que no coincidieron con nadie: ranura libre o una nueva.
        for landmarks, medida in pendientes:
            persona = next((p for p in self.people if not p.seen), None)
            if persona is None:
                persona = self._nueva()
            if persona is None:
                break                      # ya se alcanzo el maximo
            persona.reset()                # es alguien distinto al de antes
            self._asignar(persona, landmarks, medida)

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

    def _asignar(self, persona, landmarks, medida):
        cx, cy, size = medida
        if persona.landmarks is not None:
            # Velocidad suavizada: sirve para predecir donde va a estar en
            # el cuadro siguiente y no perderla cuando se mueve rapido.
            vx = cx - persona.center[0]
            vy = cy - persona.center[1]
            persona.velocity = (persona.velocity[0] * VEL_SMOOTH + vx * (1 - VEL_SMOOTH),
                                persona.velocity[1] * VEL_SMOOTH + vy * (1 - VEL_SMOOTH))
        persona.landmarks = landmarks
        persona.center = (cx, cy)
        if size > 0.01:
            # La altura se promedia: un cuadro con los pies mal detectados
            # no debe cambiar de golpe la referencia de tamano.
            persona.size = size if persona.size <= 0.01 else persona.size * 0.8 + size * 0.2
        persona.seen = True

