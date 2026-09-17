"""Del resultado crudo de MediaPipe a un esqueleto listo para dibujar.

MediaPipe entrega 33 puntos normalizados (0..1). Aqui se convierten a
pixeles del lienzo de salida y se calculan las medidas derivadas que el
render necesita: centro de hombros, centro de cadera, radio y giro de la
cabeza, escala del cuerpo y que lado esta mas cerca de la camara.
"""

import math

import numpy as np

# Indices de los 33 landmarks de MediaPipe Pose.
NOSE = 0
L_EYE, R_EYE = 2, 5
L_EAR, R_EAR = 7, 8
MOUTH_L, MOUTH_R = 9, 10
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_INDEX, R_INDEX = 19, 20
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_FOOT, R_FOOT = 31, 32

# Huesos que se dibujan como extremidades: (inicio, fin, nombre logico).
BONES_LEFT = [
    (L_SHOULDER, L_ELBOW, "upper_arm"),
    (L_ELBOW, L_WRIST, "forearm"),
    (L_HIP, L_KNEE, "thigh"),
    (L_KNEE, L_ANKLE, "shin"),
]
BONES_RIGHT = [
    (R_SHOULDER, R_ELBOW, "upper_arm"),
    (R_ELBOW, R_WRIST, "forearm"),
    (R_HIP, R_KNEE, "thigh"),
    (R_KNEE, R_ANKLE, "shin"),
]


def _mid(a, b):
    return (a + b) * 0.5


class Skeleton:
    """Esqueleto en pixeles con todo lo derivado ya calculado."""

    def __init__(self, pts, vis, z, width, height):
        self.pts = pts              # (33, 2) en pixeles
        self.vis = vis              # (33,) visibilidad 0..1
        self.z = z                  # (33,) profundidad relativa
        self.width = width
        self.height = height

        self.shoulder_c = _mid(pts[L_SHOULDER], pts[R_SHOULDER])
        self.hip_c = _mid(pts[L_HIP], pts[R_HIP])

        self.shoulder_w = float(np.linalg.norm(pts[L_SHOULDER] - pts[R_SHOULDER]))
        self.torso_len = float(np.linalg.norm(self.shoulder_c - self.hip_c))

        # Escala global del cuerpo. Se toma el mayor de dos medidas porque
        # de frente domina el ancho de hombros y de perfil domina el torso.
        self.scale = max(self.shoulder_w, self.torso_len * 0.75, 1e-3)

        self.head_c, self.head_r, self.head_angle = self._head()

        # Angulo del torso: de cadera hacia hombros.
        d = self.shoulder_c - self.hip_c
        self.torso_angle = math.degrees(math.atan2(d[1], d[0]))

        # Que lado esta mas cerca de la camara (z menor = mas cerca).
        self.left_is_near = float(z[L_SHOULDER] + z[L_HIP]) < float(z[R_SHOULDER] + z[R_HIP])

    def _head(self):
        pts, vis = self.pts, self.vis
        nose = pts[NOSE]

        # Radio: se combinan tres estimaciones y se toma la mayor confiable.
        candidates = [0.34 * self.shoulder_w]
        if vis[L_EAR] > 0.4 and vis[R_EAR] > 0.4:
            ear_d = float(np.linalg.norm(pts[L_EAR] - pts[R_EAR]))
            candidates.append(0.72 * ear_d)
        neck_d = float(np.linalg.norm(nose - self.shoulder_c))
        candidates.append(0.52 * neck_d)
        radius = float(np.clip(max(candidates), 8.0, self.height * 0.45))

        # Centro: la nariz esta al frente de la cara, el centro del craneo
        # queda desplazado hacia atras/arriba respecto de ella.
        ear_c = _mid(pts[L_EAR], pts[R_EAR]) if (vis[L_EAR] > 0.4 and vis[R_EAR] > 0.4) else nose
        center = _mid(nose, ear_c)
        up = center - self.shoulder_c
        n = float(np.linalg.norm(up))
        if n > 1e-3:
            center = center + (up / n) * (radius * 0.22)

        # Giro: la linea entre orejas u ojos define la inclinacion.
        if vis[L_EAR] > 0.4 and vis[R_EAR] > 0.4:
            d = pts[L_EAR] - pts[R_EAR]
        else:
            d = pts[L_EYE] - pts[R_EYE]
        angle = math.degrees(math.atan2(d[1], d[0])) if float(np.linalg.norm(d)) > 1e-3 else 0.0
        return center, radius, angle

    def visible(self, idx, thr=0.35):
        return float(self.vis[idx]) > thr

    def bbox(self, margin_ratio=0.6):
        """Caja que encierra al avatar, con margen para grosor, borde y halo.

        El compositor la usa para mezclar solo donde hay algo que mezclar.
        El margen es generoso a proposito: recortar de mas se ve como un
        avatar con los bordes cortados, y ahorrar esos pixeles no vale eso.
        """
        usable = self.pts[self.vis > 0.3]
        if usable.shape[0] == 0:
            usable = self.pts
        m = self.scale * margin_ratio
        x0 = float(np.min(usable[:, 0])) - m
        y0 = float(np.min(usable[:, 1])) - m
        x1 = float(np.max(usable[:, 0])) + m
        y1 = float(np.max(usable[:, 1])) + m
        return (max(int(x0), 0), max(int(y0), 0),
                min(int(x1) + 1, self.width), min(int(y1) + 1, self.height))

    def bone(self, a, b):
        return self.pts[a], self.pts[b]


def from_landmarks(landmarks, width, height, smoother=None, fps=None):
    """Convierte los landmarks normalizados de MediaPipe en un Skeleton.

    landmarks: lista de 33 NormalizedLandmark del PoseLandmarkerResult.
    """
    n = len(landmarks)
    pts = np.empty((n, 2), dtype=np.float32)
    vis = np.empty(n, dtype=np.float32)
    z = np.empty(n, dtype=np.float32)

    for i, lm in enumerate(landmarks):
        pts[i, 0] = lm.x * width
        pts[i, 1] = lm.y * height
        z[i] = lm.z
        # En algunas versiones visibility puede venir en None.
        v = getattr(lm, "visibility", None)
        vis[i] = 1.0 if v is None else float(v)

    if smoother is not None:
        pts = smoother(pts, freq=fps)

    return Skeleton(pts, vis, z, width, height)
