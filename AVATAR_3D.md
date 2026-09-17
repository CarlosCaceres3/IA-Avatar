# Camino 3D: avatar anime con VRoid + Warudo

Guía para el stand cuando se quiere un **avatar 3D estilo anime** en vez del
muñeco articulado en 2D de `avatar_cam.py`.

Este camino **no usa el código de este repositorio**. Es un stack distinto y para
"anime de verdad" da mejor resultado que dibujar piezas 2D a mano. Lo único que se
reaprovecha es la cámara virtual de OBS, que ya quedó instalada y probada.

---

## Qué hace cada programa

| Programa | Para qué | Costo | Canal |
|----------|----------|-------|-------|
| **VRoid Studio** | Crear el personaje anime 3D | Gratis | winget (pixiv) — ya instalado |
| **Warudo** | Animarlo con la webcam y mostrarlo | Gratis | Steam, app 2079120 |
| **OBS Virtual Camera** | Enviarlo al proyector / Zoom | Gratis | Ya instalado y verificado |

Hardware de este equipo: RTX 3050 6 GB + 16 GB RAM. Sobra para 3D en tiempo real.

---

## Paso 1 — Crear el personaje (VRoid Studio)

1. Abrir **VRoid Studio** → *Crear nuevo modelo*.
2. Elegir base (masculina / femenina / neutra) y ajustar cara, pelo, ropa y cuerpo.
   No hace falta saber dibujar: todo es con deslizadores y plantillas.
3. Exportar: *Menú → Exportar → Exportar como VRM*.
   - Guardar el `.vrm` en una carpeta a mano.
   - En las opciones de exportación se puede reducir el número de polígonos y de
     materiales. Para un stand conviene bajarlos: menos carga y más FPS.

**Atajo si no hay tiempo:** en [VRoid Hub](https://hub.vroid.com) hay modelos
gratuitos. Revisar la licencia de cada uno antes de usarlo en público — muchos
permiten uso libre, otros no. La licencia está en la ficha del modelo.

> Igual que con los futbolistas: un personaje anime **existente** (de una serie
> conocida) tiene dueño. Uno creado en VRoid es original y no da problemas.

---

## Paso 2 — Animarlo (Warudo)

1. Instalar **Warudo** desde Steam (gratis).
2. Abrirlo → cargar el `.vrm` exportado (*Character → Load Character*).
3. Activar el seguimiento por webcam: *Motion Capture → Face Tracking* (y
   *Hand Tracking* si se quiere que mueva las manos).
4. Ajustar la cámara de escena y el fondo.

Warudo trae seguimiento de cara y manos integrado, así que no hace falta nada más.

---

## Paso 3 — Sacarlo al proyector

Dos opciones:

- **Directo:** poner Warudo en pantalla completa en el proyector. Lo más simple.
- **Por cámara virtual:** capturar Warudo desde OBS (*Captura de ventana* o *Captura
  de juego*) y activar *Iniciar cámara virtual*. Así aparece como webcam en Zoom,
  Meet o cualquier programa — igual que hace `avatar_cam.py`.

La segunda opción permite además ponerle fondos, logos y rótulos encima con OBS.

---

## Qué se pierde y qué se gana frente a `avatar_cam.py`

**Se gana:** personaje 3D real con volumen, seguimiento facial fino (parpadeo,
boca, cejas), pelo y ropa con física, y calidad anime que no se consigue dibujando
piezas 2D.

**Se pierde:** los fondos generados, los packs de color, el cambio de avatar con una
tecla, el diagnóstico previo al evento y el arranque de un solo comando. Todo eso hay
que rehacerlo dentro de Warudo u OBS.

**Sigue sirviendo:** `avatar_cam.py` como plan B. Arranca en segundos, no depende de
Steam ni de GPU, y si algo falla en el stand se levanta con un doble clic en
`INICIAR.bat`. Para un evento en vivo, tener un plan B que ya funciona vale bastante.

---

## Prueba antes del evento

Igual que con la app propia, conviene probar el día anterior:

- ¿Cuántos FPS da Warudo en el equipo del stand, con el proyector conectado?
- ¿A qué distancia deja de seguir la cara? (marcar el piso).
- ¿Cómo se ve con la luz real del lugar? El seguimiento facial necesita luz en la
  cara, igual que la app propia.
- ¿Qué pasa cuando **no hay nadie** frente a la cámara? Conviene dejar el personaje
  en una pose de espera que se vea bien, porque es el estado que más tiempo va a
  estar en pantalla durante el evento.
