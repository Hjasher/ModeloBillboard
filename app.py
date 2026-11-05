import os, json, joblib
import pandas as pd
import streamlit as st
from openai import OpenAI

# ------------------------------------------------------------------
# Cambie aquí su nombre, aparece en el título de la app
# ------------------------------------------------------------------
NOMBRE = 'Su Nombre'

st.set_page_config(page_title='Billboard Top 1', layout='centered')
st.title(f'Predicción de Top 1 en Billboard-{NOMBRE}-ISC-Campus Comayagua')


@st.cache_resource
def cargar_modelo():
    pkl = 'arbol_billboard.pkl' if os.path.exists('arbol_billboard.pkl') else 'arbol_billboard(2).pkl'
    meta = 'model_metadata.json' if os.path.exists('model_metadata.json') else 'model_metadata(2).json'

    modelo = joblib.load(pkl)

    with open(meta, encoding='utf-8') as f:
        metadata = json.load(f)

    return modelo, metadata


@st.cache_data
def cargar_base():
    csv = 'billboard_limpio.csv' if os.path.exists('billboard_limpio.csv') else 'billboard.csv'
    return pd.read_csv(csv)


modelo, metadata = cargar_modelo()
df = cargar_base()

opciones = metadata['opciones']
clases = {int(k): v for k, v in metadata['clases'].items()}

st.caption(metadata['nombre_modelo'])

with st.form('datos'):

    col1, col2 = st.columns(2)

    artist_type = col1.selectbox('Tipo de artista', opciones['artist_type'])
    genre = col2.selectbox('Género', opciones['genre'])

    mood = col1.selectbox('Estado de ánimo (mood)', opciones['mood'])
    tempo = col2.selectbox('Tempo', opciones['tempo'])

    duracion_min = int(metadata['durationSeg_min'])
    duracion_max = int(metadata['durationSeg_max'])
    duracion_base = int(df['durationSeg'].median())

    durationSeg = st.number_input(
        'Duración de la canción (segundos)',
        duracion_min,
        duracion_max,
        min(max(duracion_base, duracion_min), duracion_max)
    )

    enviar = st.form_submit_button('Evaluar')

if enviar:

    cancion = pd.DataFrame([{
        'artist_type': artist_type,
        'tempo': tempo,
        'mood': mood,
        'genre': genre,
        'durationSeg': durationSeg
    }])

    # Mismo procesamiento del notebook, One-Hot Encoding y mismas columnas del entrenamiento
    cancion_dummies = pd.get_dummies(cancion)
    cancion_procesada = cancion_dummies.reindex(
        columns=metadata['columnas_entrenamiento'],
        fill_value=0
    ).astype(float)

    prediccion = int(modelo.predict(cancion_procesada)[0])
    probabilidades = modelo.predict_proba(cancion_procesada)[0]
    prob_top1 = float(probabilidades[list(modelo.classes_).index(1)])

    resultado = clases.get(prediccion, 'No definido')

    st.subheader(f'Resultado: {resultado}')
    st.write(f'Probabilidad estimada de llegar al top 1: {prob_top1:.1%}')
    st.progress(min(max(prob_top1, 0.0), 1.0))

    api_key = st.secrets.get(
        'GROQ_API_KEY',
        os.getenv('GROQ_API_KEY', '')
    )

    if api_key:

        prompt = f'''
        Actúa como analista de la industria musical.

        Explica brevemente el resultado y brinda
        3 recomendaciones prudentes.

        Datos de la canción:
        tipo de artista={artist_type}
        género={genre}
        mood={mood}
        tempo={tempo}
        duración (segundos)={durationSeg}

        Resultado del modelo:
        predicción={resultado}
        probabilidad de top 1={prob_top1:.1%}
        '''

        try:

            client = OpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1"
            )

            respuesta = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {
                        "role": "system",
                        "content": "Eres un asesor profesional de la industria musical."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            texto = respuesta.choices[0].message.content

            st.info(texto)

        except Exception as e:

            st.warning(f'Error con Groq: {e}')

    else:

        st.warning(
            'Agregue GROQ_API_KEY en los secretos de Streamlit.'
        )

st.divider()

st.write('Vista rápida de la base principal')

st.dataframe(
    df.head(20),
    width='stretch'
)
