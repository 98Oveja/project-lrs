
from tensorflow.keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input

#streamlit
import streamlit as st
import mediapipe as mp
import numpy as np
import cv2



mp_holistic = mp.solutions.holistic #modelo de mp
mp_drawing = mp.solutions.drawing_utils #importando utilidades
# Path del modelo preentrenado
MODEL_PATH = 'models/modelFrases.h5'
# Cargamos el modelo preentrenado
model = load_model(MODEL_PATH)
actions = np.array(['por favor','feliz','mucho gusto','perdóname','hola','adiós','gracias','yo','ayuda'])


# -------  generar deteccion mediapipe ---------
def mediapipe_detection(image, model):
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # CONVERSIÓN DE COLOR BGR 2 RGB
    image.flags.writeable = False                  # La imagen ya no se puede escribir, por eso es false
    results = model.process(image)                 # realizar prediction
    image.flags.writeable = True                   # ahora se puede escribir en la img
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # conversion de color RGB 2 BGR
    return image, results

# --------- formateo de las marcas medipipe ---------
def draw_formateado_landmarks(image, results):
    # dibujar conexiones de cara
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
            mp_drawing.DrawingSpec(color=(0,255,255), thickness=1, circle_radius=1), 
            mp_drawing.DrawingSpec(color=(255,51,51), thickness=1, circle_radius=1)
            ) 
    # dibujar conexiones de poses
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
            mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
            ) 
    # dibujar conexiones de mano izquierda
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
            mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
            mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
            ) 
    # dibuajr conexiones de mano derecha
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            ) 
# ----------- Extraer los keypoints ------------
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def main():
    st.title("Sistema de Reconocimiento de Lengua de Señas (LENSEGUA) 🇬🇹")
    iniciar = st.button('INICIAR')
    stframe = st.image([])
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Traducción**")
        col1_text = st.markdown("...")
    with col2:
        st.markdown("**Puntos**")
        col2_text = st.markdown("0")
    with col3:
        st.markdown("**% predicción**")
        col3_text = st.markdown("0.3")
    st.markdown("<hr/>", unsafe_allow_html=True)
    
    vid = cv2.VideoCapture(0)

    secuencia =[]
    sentencia = []
    predicciones = []
    threshold = 0.3
    predic_porcent =0
    traduction = ''

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while iniciar:
            ret, frame = vid.read()
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image, results = mediapipe_detection(frame, holistic)
            draw_formateado_landmarks(image, results)
            stframe.image(image)

            keypoints = extract_keypoints(results)
            secuencia.append(keypoints)
            secuencia = secuencia[-30:] #ultimos 30 puntos clave

            if len(secuencia) == 30:
                resultado = model.predict(np.expand_dims(secuencia, axis=0))[0]
                print(actions[np.argmax(resultado)])
                predicciones.append(np.argmax(resultado))

                if np.unique(predicciones[-10:])[0]==np.argmax(resultado): 
                    if resultado[np.argmax(resultado)] > threshold: 
                        predic_porcent=resultado[np.argmax(resultado)]
                        if len(sentencia) > 0: 
                            if actions[np.argmax(resultado)] != sentencia[-1]:
                                sentencia.append(actions[np.argmax(resultado)])
                                traduction = actions[np.argmax(resultado)]

                        else:
                            sentencia.append(actions[np.argmax(resultado)])
                            traduction = actions[np.argmax(resultado)]
                        
                    
                if len(sentencia) > 5:
                    sentencia = sentencia[-5:]
            
            col1_text.write(f'<p style="font-size:18px; text-align:center">{traduction}</p>',unsafe_allow_html=True)
            col3_text.write(f'<p style="font-size:18px; text-align:center">{int(predic_porcent*100)}%</p>',unsafe_allow_html=True)
                

        else:
            st.write('Detenido')


if __name__ == "__main__":
    main()
