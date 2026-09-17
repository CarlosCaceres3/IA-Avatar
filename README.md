# Avatar IA en vivo — demo para stand

Una persona se para frente a la cámara y en pantalla aparece un avatar articulado
que copia sus movimientos en tiempo real. La persona real no se ve: solo el avatar
sobre el fondo elegido. La salida se puede enviar a una **cámara virtual** para que
Zoom, Meet, OBS o el proyector la tomen como si fuera una webcam más.

Todo corre **local, en CPU y sin internet** una vez descargado el modelo.

> ¿Buscando un avatar **3D estilo anime** en vez del muñeco 2D? Ese es otro stack
> (VRoid + Warudo) y está documentado en [AVATAR_3D.md](AVATAR_3D.md). Esta app
> sigue sirviendo como plan B: arranca en segundos y no depende de GPU ni de Steam.

---

## Instalación

Requiere Python 3.9–3.13.

```bash
pip install -r requirements.txt
```

El modelo de pose (9 MB) no viene en el repositorio, se descarga aparte:

```bash
python tools/descargar_modelo.py
```

## Arranque rápido

**Doble clic en `INICIAR.bat`.** Arranca en pantalla completa y no necesita nada más.
Si falta el modelo lo descarga solo, y si faltan las librerías las instala.

**Doble clic en `DIAGNOSTICO.bat`** para el chequeo previo al evento: paquetes,
modelo, cámaras, FPS reales y cámara virtual.

Ya hay accesos directos en el escritorio: **Avatar IA** y **Avatar IA - Chequeo**.
Si se pierden, se recrean con clic derecho sobre cada `.bat` → *Enviar a* →
*Escritorio (crear acceso directo)*.

Desde una terminal, si prefieres:

```bash
python avatar_cam.py --fullscreen
python tools/diagnostico.py
```

### Opciones del stand

Se editan en la línea del final de `INICIAR.bat`, o se pasan por terminal:

| Opción | Para qué |
|--------|----------|
| `--avatar N` | con cuál arranca (0 a 10) |
| `--personas N` | cuántas personas a la vez (1 a 4) |
| `--deteccion 0.4` | si la segunda persona no aparece |
| `--calidad 0.5` | más rápido si el equipo va justo |
| `--camera 1` | si usa otra cámara |
| `--exposure -5` | sube el FPS, pero solo con mucha luz |

---

## Varias personas a la vez

Por defecto sigue hasta **2 personas**, y le da a cada una **un avatar distinto**
para que se distingan. Las teclas `A` / `D` cambian los avatares de todas a la vez.

```bash
python avatar_cam.py --personas 3      # hasta 3 (1 a 4)
python avatar_cam.py --personas 1      # una sola, lo mas rapido
```

Cada persona lleva su propio filtro de suavizado y su propio tamaño de cuerpo.

**Por qué no basta con pedirle varias poses a MediaPipe:** el detector no garantiza
devolverlas siempre en el mismo orden, así que asignar el avatar por posición en la
lista haría que dos personas se intercambiaran los avatares constantemente. En cada
cuadro las poses se emparejan con las personas ya conocidas por **cercanía**, por
**hacia dónde venían moviéndose** y por su **altura en el cuadro**, así que quien ya
estaba conserva el suyo. Si alguien desaparece un instante, su avatar lo espera 12
cuadros antes de liberarse.

**Si la segunda persona no se detecta**, hay dos perillas:

```bash
python avatar_cam.py --deteccion 0.4        # menos exigente al detectar
python avatar_cam.py --ancho-deteccion 800  # mas pixeles por persona
```

Bajar `--deteccion` ayuda cuando la segunda persona sale parcialmente tapada; subirla
evita detecciones fantasma. El ancho de detección cuesta poco: medido en este equipo,
la detección tarda 11-13 ms entre 480 y 960 px, porque MediaPipe reescala a su tamaño
interno igual. El valor por defecto ya es 640.

El contador `Personas: 1/2` del HUD dice en todo momento a cuántas está viendo: es la
forma rápida de ajustar estas dos perillas en el sitio.

**Costo medido** en este equipo, a 720p. Los números se tomaron con el sombreado de
volumen activo, que era el caso más pesado; con los avatares actuales sobra margen:

| Personas | Por cuadro | FPS teóricos |
|----------|-----------|--------------|
| 1 | 13.0 ms | 77 |
| 2 | 23.4 ms | 43 |
| 3 | 35.3 ms | 28 |

Con dos va sobrado. Si necesitan tres o cuatro y el equipo va justo, bajen la calidad
con `--calidad 0.5`.

---

## Teclas

| Tecla | Acción |
|-------|--------|
| `A` / `D` | Avatar anterior / siguiente |
| `F` | Cambiar fondo |
| `C` | Cara real de la persona sobre el avatar |
| `G` | Espejo on/off |
| `E` | Mostrar el esqueleto detectado (depuración) |
| `H` | Ocultar/mostrar los datos en pantalla |
| `V` | Activar/pausar la cámara virtual |
| `P` | Guardar una foto PNG en `capturas/` |
| `TAB` | Pantalla completa |
| `Q` o `ESC` | Salir |

---

## Cómo funciona

```
webcam ──► PoseLandmarker ──► 33 puntos por persona ──► One Euro
                                        │
                        reparto por persona (cada una su avatar)
                                        │
                     renderizador ──► fondo + halo ──► ventana + cámara virtual
```

El detector corre **en su propio hilo** y el bucle principal nunca lo espera: toma
el último resultado disponible.

- **`src/skeleton.py`** convierte los 33 puntos normalizados de MediaPipe en
  píxeles y calcula lo derivado: centro de hombros y cadera, radio y giro de la
  cabeza, escala del cuerpo y qué lado está más cerca de la cámara.
- **`src/smoothing.py`** aplica un filtro One Euro. Sin él, el avatar vibra 1–3 px
  aunque la persona esté quieta, y en proyección grande eso se nota mucho.
- **`src/avatar.py`** dibuja el cuerpo de atrás hacia adelante (pierna lejana,
  brazo lejano, torso, pierna cercana, brazo cercano, cabeza) para que las partes
  se tapen en el orden correcto.
- **`src/people.py`** reparte las poses detectadas entre personas estables, para que
  cada una conserve su avatar entre cuadros.
- **`src/stage.py`** genera el fondo, pega el avatar y le agrega el halo.
- **`src/camera_out.py`** envía el resultado a la cámara virtual.

---

## Avatares

Cada avatar es una carpeta en `assets/packs/`. Se cambian en vivo con `A` y `D`.

### Futbolista configurable

```bash
python tools/crear_pack_futbolista.py --nombre "Futbolista Verde" \
    --camiseta "#1B7F3B" --pantaloneta "#FFFFFF" --medias "#1B7F3B" --numero 9
```

Genera un futbolista completo con camiseta, número al pecho, pantaloneta, medias y
botines en los colores que le pasen. Cada llamada crea una carpeta nueva, así que
pueden tener varios equipos y alternarlos durante el evento.

Opciones: `--camiseta --pantaloneta --medias --botines --piel --pelo --numero --nombre`.

### Sombreado con volumen (desactivado)

El motor sabe sombrear cada extremidad como un cilindro y la cabeza como una esfera
(luz, brillo, contorno y sombra en el piso). Ningún avatar lo usa por defecto: los
temas que lo activaban se quitaron a pedido.

Para recuperarlo, basta agregar un `Theme` en `src/avatar.py` con `volume=True`:

```python
Theme(
    name="3D Azul",
    suit=(190, 105, 45), skin=(150, 190, 225), accent=(225, 165, 80),
    outline=(40, 28, 20), glow=(200, 120, 50), glow_strength=0.30,
    outline_w=0.016, volume=True, rim=(255, 225, 180),
),
```

El código está en `src/shading.py`. Cuesta unos 22 ms por cuadro a 720p (contra 3 ms
del dibujo plano), por eso se dibuja a 0.6 de resolución y se amplía al componer.

### Cara real de la persona (tecla `C`)

Pulsando **`C`** el avatar deja de tener cara dibujada y lleva **la cara de quien
está frente a la cámara**, recortada en óvalo con borde difuminado y siguiendo la
inclinación de su cabeza. Funciona con cualquier avatar y con las dos personas a la
vez: cada una lleva la suya.

Es la forma de poner "una persona real" sin problemas de derechos de imagen — la
cara es la de quien está ahí, por voluntad propia. Y para un stand suele funcionar
mejor que un personaje famoso: la gente se ve **a sí misma** convertida en personaje.

No necesita detector facial ni modelo extra: el esqueleto ya sabe dónde está la
cabeza, cuánto mide y cuánto está inclinada, y el avatar se dibuja con esa misma
inclinación, así que la cara del cuadro ya viene orientada. Está en `src/realface.py`
y se ajusta con `GAIN` (tamaño), `FEATHER` (suavidad del borde) y `ASPECT` (forma).

### Personajes de dominio público

```bash
python tools/crear_packs_historicos.py
```

Genera tres avatares de personajes **libres de derechos**: **Bolívar** (casaca azul,
banda roja, charreteras y su peinado), **Quijote** (armadura, yelmo de bacía y barba
en punta) y **Frankenstein** (cabeza plana, flequillo y pernos).

Las piezas se dibujan por código, así que tampoco dependen de ilustraciones de
terceros: el pack completo es original y se puede usar sin pedir permiso a nadie.

> **Sobre famosos:** la cara de una persona viva y reconocible —foto o caricatura—
> está protegida por su derecho de imagen, y una caricatura se define justamente por
> ser reconocible. Las alternativas que sí funcionan: la cara del propio visitante
> (tecla `C`), alguien que dé permiso por escrito, personajes de dominio público como
> estos tres, o un personaje con licencia comprada.

### Poner sus propias imágenes

```bash
python tools/crear_pack_ejemplo.py     # crea un pack de muestra para copiar
```

**Reemplacen los PNG por sus dibujos manteniendo el nombre** y el avatar los usa,
sin tocar código. Archivos que reconoce (todos opcionales — lo que falte se dibuja
con figuras del tema):

```
head.png  torso.png  upper_arm.png  forearm.png  thigh.png  shin.png  hand.png  foot.png
```

Para una pieza distinta en cada lado: `forearm_l.png`, `forearm_r.png`.

**Lo único que hay que respetar:**

- Fondo transparente (canal alfa de verdad, no blanco).
- En los huesos (brazos, piernas, torso), el dibujo va **de arriba hacia abajo**:
  arriba la articulación que queda del lado del cuerpo, abajo la del extremo.
- Mínimo unos 200 px de alto por pieza, o se verá pixelado en el proyector.

**No hace falta recortar al píxel.** Las articulaciones se deducen solas del canal
alfa: el programa mide dónde empieza y termina el dibujo dentro del PNG e ignora el
margen transparente. Pueden exportar con el espacio que les quede cómodo.

Revisen el resultado sin pararse frente a la cámara:

```bash
python tools/probar_pack.py mi_personaje
```

Avisa qué partes faltan, detecta los errores típicos (sin transparencia, imagen muy
chica, pieza mal orientada) y genera `pack_revision.png` con el avatar en 4 poses.

### theme.json

Los colores y ajustes van en `theme.json` dentro de la carpeta del pack:

```json
{
  "name": "Mi personaje",
  "skin": "#C68642",
  "suit": "#D32F2F",
  "accent": "#D32F2F",
  "outline": "#2A2520",
  "glow": "#D32F2F",
  "glow_strength": 0.35,
  "draw_face": true,
  "parts": {
    "head": { "size": 1.30 }
  }
}
```

- `skin` cubre las partes sin PNG (por ejemplo el cuello): pónganla en el tono del pack.
- `draw_face: true` hace que el programa pinte los ojos y la boca **encima** de
  `head.png`. Si su dibujo ya trae cara, pónganlo en `false`.
- `parts` es opcional y sirve para retocar: `size` agranda la pieza, `width` la
  ensancha, y `a` / `b` / `anchor` permiten fijar las articulaciones a mano si la
  detección automática no acierta. Lo que declaren se combina con lo detectado.

### Sobre usar personas reales

Si piensan usar la cara de alguien conocido, tengan en cuenta que eso toca sus
derechos de imagen y, si la foto es de prensa, también los derechos del fotógrafo.
Alternativas que funcionan igual de bien para un stand: personajes propios, el
futbolista configurable de arriba, personajes de dominio público, o la mascota de
la marca si tienen una.

**Fondos:** cualquier `.jpg` o `.png` que dejen en `assets/backgrounds/` aparece en
el ciclo de la tecla `F`.

---

## Cámara virtual (para el proyector, Zoom u OBS)

**Ya está instalada y verificada en este equipo** (OBS Studio 32.2.1). La app detecta
*OBS Virtual Camera* automáticamente al arrancar y lo confirma en pantalla:
`VirtualCam: OBS Virtual Camera`. Desde ahí la salida aparece como una webcam más en
Zoom, Meet, Teams, OBS o cualquier programa que liste cámaras.

Probado leyendo la salida desde otro proceso: llega **1280×720 exacto**, y el
consumidor puede pedir la resolución que quiera (640×480, 720p o 1080p) — OBS la
negocia del lado de quien recibe.

Para montarlo en **otro** equipo:

```bash
winget install --id OBSProject.OBSStudio -e
```

No hace falta abrir OBS: el instalador registra el driver solo.

Si el driver no está, **la app igual funciona**: muestra la ventana normal y avisa en
pantalla que la cámara virtual no está disponible. Para el stand, la ventana en
pantalla completa (`TAB`) puede bastar.

### Cómo se genera el video, técnicamente

La cámara virtual no graba un archivo: publica cada cuadro ya renderizado en un
dispositivo de video del sistema operativo. El bucle es:

```python
cam = pyvirtualcam.Camera(width=1280, height=720, fps=30, fmt=PixelFormat.BGR)
...
out = stage.composite(background, canvas, ...)   # el cuadro final, un array BGR
cam.send(out)                                    # se publica como webcam
cam.sleep_until_next_frame()                     # mantiene el ritmo de 30 FPS
```

Se usa `PixelFormat.BGR` porque es el formato nativo de OpenCV: así no hay que
convertir el color en cada cuadro.

Si más adelante quieren **grabar a archivo** en vez de (o además de) transmitir, es
el mismo array y tres líneas:

```python
writer = cv2.VideoWriter("clip.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
writer.write(out)    # dentro del bucle
writer.release()     # al terminar
```

---

## Rendimiento

Medido en este equipo, a 1280×720:

| Etapa | Costo por cuadro |
|-------|------------------|
| Detección de pose (hilo aparte) | 10.8 ms |
| Render del avatar | 2.3 ms |
| Fondo + halo + composición | 7.3 ms |

El límite real termina siendo **la webcam**, no el procesamiento.

### El detalle de la exposición — y su trampa

Con exposición automática y poca luz, la cámara alarga el tiempo de cada toma y el
FPS se desploma. Fijarla a mano lo arregla… **pero solo si hay luz suficiente.**

Medido en este equipo, en una sala con poca luz:

| Exposición | FPS | Personas detectadas |
|------------|-----|---------------------|
| Automática | 12.4 | 27 de 40 |
| Manual `-5` | 27.1 | **0 de 40** |

Con `-5` la demo corre al doble de velocidad y **no detecta absolutamente a nadie**,
porque la imagen queda casi negra y el detector no encuentra un cuerpo ahí. Unos FPS
altos con cero detecciones se ven en pantalla como una app "fluida" que no reacciona.

Por eso la exposición manual **no viene activada por defecto**. Si el stand está bien
iluminado, arranquen con:

```bash
python avatar_cam.py --exposure -5
```

y comprueben con el diagnóstico que siguen detectando personas:

```bash
python tools/diagnostico.py --exposure -5
```

Si la cuenta de "cuadros con persona detectada" baja, falta luz: suban el valor
(`-4`, `-3`) o quiten la opción. **Prueben esto con la luz real del stand antes de
abrir**, porque cambia el resultado por completo.

> Ojo: la exposición manual **queda grabada en el driver de la cámara y sobrevive al
> cierre del programa**. Si no se restaura, la webcam sigue oscura para la siguiente
> aplicación que la use (y el detector deja de encontrar personas). `avatar_cam.py` y
> el diagnóstico la devuelven a automático al salir; si alguna vez matan el proceso a
> la fuerza y la cámara queda oscura, se arregla abriéndola una vez sin `--exposure`.

### Si aún va lento

1. Bajen la resolución: `--width 960 --height 540`.
2. Usen el modelo liviano:
   ```bash
   python tools/descargar_modelo.py --todos
   python avatar_cam.py --model models/pose_landmarker_lite.task
   ```
3. Elijan un avatar sin halo (el tema *Cartoon* tiene `glow_strength` en 0).

---

## Opciones de línea de comandos

```
--camera N        índice de la webcam (por defecto 0)
--width / --height  resolución de captura (1280x720)
--fps N           FPS objetivo (30)
--exposure V      exposición manual, ej. -5
--model RUTA      archivo .task del modelo de pose
--avatar N        avatar inicial
--no-virtualcam   no abrir la cámara virtual
--fullscreen      arrancar en pantalla completa
--max-frames N    salir tras N cuadros (para probar)
--headless        no abrir ventana (para verificar sin pantalla)
```

---

## Problemas frecuentes

**"No pude abrir la cámara 0"** — otra aplicación la está usando (Teams, Zoom,
el navegador). Ciérrala, o prueba `--camera 1`.

**El avatar no aparece** — la persona debe verse de cuerpo completo o al menos de
medio cuerpo. A menos de metro y medio de la cámara el detector pierde las piernas.
Presiona `E` para ver si el esqueleto se está detectando.

**El avatar tiembla** — sube el suavizado bajando `min_cutoff` en la creación del
`OneEuroFilter` en `avatar_cam.py` (por ejemplo de `1.1` a `0.7`). A cambio, el
movimiento rápido se siente un poco más lento.

**El avatar va con retraso** — es lo contrario: sube `min_cutoff`.

**Se ve todo espejado al revés** — presiona `G`.

