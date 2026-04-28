import cv2
import mediapipe as mp
import math
import time
import os

CONEXIONES_MANO = [
    (0, 1), (1, 2), (2, 3), (3, 4),       
    (0, 5), (5, 6), (6, 7), (7, 8),       
    (5, 9), (9, 10), (10, 11), (11, 12),  
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) 
]

def dibujar_esqueleto(frame, hand_landmarks, w, h):
    for conexion in CONEXIONES_MANO:
        p1, p2 = hand_landmarks[conexion[0]], hand_landmarks[conexion[1]]
        cv2.line(frame, (int(p1.x * w), int(p1.y * h)), (int(p2.x * w), int(p2.y * h)), (255, 255, 255), 2)
    for mark in hand_landmarks:
        cv2.circle(frame, (int(mark.x * w), int(mark.y * h)), 4, (0, 215, 255), cv2.FILLED)

# --- FUNCIONES DE GESTOS ---
def es_gesto_dos_dedos(mano):
    dist = math.hypot(mano[8].x - mano[12].x, mano[8].y - mano[12].y)
    return dist < 0.05 

def es_pulgar_arriba(mano):
    punta_pulgar = mano[4].y
    return all(punta_pulgar < mano[i].y for i in [5, 9, 13, 17, 8, 12, 16, 20])

cine_img = cv2.imread('imagenes/image.png') 
like_img = cv2.imread('imagenes/like.png') 

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

hand_options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE, num_hands=2,
    min_hand_detection_confidence=0.4, min_hand_presence_confidence=0.4)

face_options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE, num_faces=1)

hand_detector = mp.tasks.vision.HandLandmarker.create_from_options(hand_options)
face_detector = mp.tasks.vision.FaceLandmarker.create_from_options(face_options)

cap = cv2.VideoCapture(0)
gif_cap = cv2.VideoCapture('imagenes/giphy.gif') 

UMBRAL_MOVIMIENTO = 0.003 
UMBRAL_ERROR = 0.15 
pos_x_anterior = None 
UMBRAL_PROXIMIDAD_CARA = 150 

TIEMPO_GRACIA_GIF = 0.7
TIEMPO_GRACIA_CINE = 0.02
TIEMPO_GRACIA_LIKE = 0.4 

ultimo_tiempo_movimiento = 0 
ultimo_tiempo_pose_cine = 0 
ultimo_tiempo_like = 0

WIN_CINE = "Es cine :<"
WIN_LIKE = "No era png :("
WIN_GIF  = "Q pro"

print("Cargo correctamente")

while cap.isOpened():
    exito, frame = cap.read()
    if not exito: continue
    frame = cv2.flip(frame, 1)
    h_cam, w_cam, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    hand_result = hand_detector.detect(mp_image)
    face_result = face_detector.detect(mp_image)

    movimiento_detectado_ahora = False
    pose_cine_detectada_ahora = False
    like_detectado_ahora = False
    
    if face_result.face_landmarks and hand_result.hand_landmarks and len(hand_result.hand_landmarks) == 2:
        nariz_pt = face_result.face_landmarks[0][1] 
        nariz_x, nariz_y = int(nariz_pt.x * w_cam), int(nariz_pt.y * h_cam)
        mano_1, mano_2 = hand_result.hand_landmarks[0], hand_result.hand_landmarks[1]

        dist_1 = math.hypot(int(mano_1[8].x * w_cam) - nariz_x, int(mano_1[8].y * h_cam) - nariz_y)
        dist_2 = math.hypot(int(mano_2[8].x * w_cam) - nariz_x, int(mano_2[8].y * h_cam) - nariz_y)

        if dist_1 < UMBRAL_PROXIMIDAD_CARA and dist_2 < UMBRAL_PROXIMIDAD_CARA:
            pose_cine_detectada_ahora = True
            ultimo_tiempo_pose_cine = time.time()
        
        elif es_pulgar_arriba(mano_1) and es_pulgar_arriba(mano_2):
            like_detectado_ahora = True
            ultimo_tiempo_like = time.time()

        elif not pose_cine_detectada_ahora and not like_detectado_ahora:
            pose_en_cara_ok = False
            mano_libre = None
            if dist_1 < dist_2 and dist_1 < UMBRAL_PROXIMIDAD_CARA and es_gesto_dos_dedos(mano_1):
                pose_en_cara_ok, mano_libre = True, mano_2
            elif dist_2 < dist_1 and dist_2 < UMBRAL_PROXIMIDAD_CARA and es_gesto_dos_dedos(mano_2):
                pose_en_cara_ok, mano_libre = True, mano_1

            if pose_en_cara_ok and mano_libre:
                pos_x_actual = mano_libre[0].x 
                if pos_x_anterior is not None:
                    diff_mov = abs(pos_x_actual - pos_x_anterior)
                    if UMBRAL_MOVIMIENTO < diff_mov < UMBRAL_ERROR:
                        ultimo_tiempo_movimiento = time.time()
                        movimiento_detectado_ahora = True
                pos_x_anterior = pos_x_actual
            else: pos_x_anterior = None 

        dibujar_esqueleto(frame, mano_1, w_cam, h_cam)
        dibujar_esqueleto(frame, mano_2, w_cam, h_cam)
    else: pos_x_anterior = None 

    tiempo_actual = time.time()
    mostrar_cine = (tiempo_actual - ultimo_tiempo_pose_cine) < TIEMPO_GRACIA_CINE
    mostrar_like = (tiempo_actual - ultimo_tiempo_like) < TIEMPO_GRACIA_LIKE
    mostrar_gif = (tiempo_actual - ultimo_tiempo_movimiento) < TIEMPO_GRACIA_GIF

    if mostrar_cine:
        status_debug = "¡ABSOLUTE CINE!"
        if cine_img is not None: cv2.imshow(WIN_CINE, cine_img)
        for w in [WIN_GIF, WIN_LIKE]: 
            try: cv2.destroyWindow(w)
            except: pass
    elif mostrar_like:
        status_debug = "¡BUENA ESA!"
        if like_img is not None: cv2.imshow(WIN_LIKE, like_img)
        for w in [WIN_GIF, WIN_CINE]:
            try: cv2.destroyWindow(w)
            except: pass
    elif mostrar_gif:
        status_debug = "¡BAILANDO!"
        ret_gif, frame_gif = gif_cap.read()
        if not ret_gif: gif_cap.set(cv2.CAP_PROP_POS_FRAMES, 0); ret_gif, frame_gif = gif_cap.read()
        if ret_gif: cv2.imshow(WIN_GIF, frame_gif)
        for w in [WIN_CINE, WIN_LIKE]:
            try: cv2.destroyWindow(w)
            except: pass
    else:
        status_debug = "Esperando gesto..."
        for w in [WIN_GIF, WIN_CINE, WIN_LIKE]:
            try: cv2.destroyWindow(w)
            except: pass

    cv2.putText(frame, status_debug, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow('Detector de Pros', frame)
    if cv2.waitKey(5) & 0xFF == 27: break
cap.release()
cv2.destroyAllWindows()