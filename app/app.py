import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from sklearn.tree import export_text

# Función para cargar y limpiar datos
def get_clean_data():
    data = pd.read_csv("data/data.csv")
    data = data.drop(['id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data

# Añadir barra lateral con controles deslizantes
def add_sidebar():
    st.sidebar.header("Mediciones de Núcleos Celulares")
    data = get_clean_data()
    slider_labels = [
        ("Radio (mm)", "radius_mean"),
        ("Textura", "texture_mean"),
        ("Perímetro (mm)", "perimeter_mean"),
        ("Área (mm²)", "area_mean"),
        ("Suavidad (índice)", "smoothness_mean"),
        ("Compacidad", "compactness_mean"),
        ("Concavidad", "concavity_mean"),
        ("Puntos cóncavos ", "concave points_mean"),
        ("Simetría", "symmetry_mean"),
        ("Dimensión fractal", "fractal_dimension_mean"),
    ]
    input_dict = {}
    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label, min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )
    return input_dict

# Escalar los valores de entrada
def get_scaled_values(input_dict):
    data = get_clean_data()
    X = data.drop(['diagnosis'], axis=1)
    scaled_dict = {}
    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value
    return scaled_dict

# Generar el gráfico radar
def get_radar_chart(input_data):
    input_data = get_scaled_values(input_data)
    categories = ['Radio', 'Textura', 'Perímetro', 'Área', 
                  'Suavidad', 'Compacidad', 'Concavidad', 
                  'Puntos Cóncavos', 'Simetría', 'Dimensión Fractal']
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[input_data['radius_mean'], input_data['texture_mean'], 
           input_data['perimeter_mean'], input_data['area_mean'], 
           input_data['smoothness_mean'], input_data['compactness_mean'],
           input_data['concavity_mean'], input_data['concave points_mean'], 
           input_data['symmetry_mean'], input_data['fractal_dimension_mean']],
        theta=categories, fill='toself', name='Valor Promedio'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True
    )
    return fig

# Hacer prediccion
def add_predictions(input_data):
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    input_array_scaled = scaler.transform(input_array)
    prediction = model.predict(input_array_scaled)
    st.subheader("Predicción de Agrupación Celular")
    st.write("La agrupación celular es:")
    if prediction[0] == 0:
        st.write("<span class='diagnosis benign'>Benigna</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malicious'>Maligna</span>", unsafe_allow_html=True)
    st.write("Probabilidad de ser benigna: ", model.predict_proba(input_array_scaled)[0][0])
    st.write("Probabilidad de ser maligna: ", model.predict_proba(input_array_scaled)[0][1])
    st.write("Esta aplicación puede asistir a profesionales médicos en realizar un diagnóstico, pero no debe ser utilizada como sustituto de un diagnóstico profesional.")

# Visualizar el árbol de decisión
def visualize_decision_tree(input_data):
        decision_tree_model = pickle.load(open("model/tree_model.pkl", "rb"))
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(decision_tree_model, filled=True, rounded=True, 
                  class_names=["Benigno", "Maligno"], 
                  feature_names=get_clean_data().drop(['diagnosis'], axis=1).columns)
        st.pyplot(fig)
        tree_rules = export_text(decision_tree_model, feature_names=list(input_data.keys()))
        st.text("Camino en el árbol de decisión:")
        st.text(tree_rules)

# Visualizar KMeans
def visualize_kmeans():
        kmeans_model = pickle.load(open("model/kmeans_model.pkl", "rb"))
        data = get_clean_data().drop(['diagnosis'], axis=1)
        scaler = pickle.load(open("model/scaler.pkl", "rb"))
        X_scaled = scaler.transform(data)
        pca = PCA(n_components=2)
        components = pca.fit_transform(X_scaled)
        cluster_centers_2D = pca.transform(kmeans_model.cluster_centers_)
        fig = go.Figure(data=go.Scatter(x=components[:, 0], y=components[:, 1],
                                        mode='markers', marker=dict(size=5, color=kmeans_model.labels_, opacity=0.5),
                                        name='Datos'),
                        layout=go.Layout(title="",
                                         xaxis=dict(title='Componente'),
                                         yaxis=dict(title='Componente')))
        fig.add_trace(go.Scatter(x=cluster_centers_2D[:, 0], y=cluster_centers_2D[:, 1],
                                 mode='markers', marker=dict(color='red', size=10, line=dict(color='black', width=2)),
                                 name='Centros de Clusters'))
        st.plotly_chart(fig)

# Función principal
def main():
    st.set_page_config(page_title="Predictor de Cáncer de mama", layout="wide", initial_sidebar_state="expanded")
    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
    input_data = add_sidebar()
    with st.container():
        st.title("Predictor de Cáncer de mama")
        st.write("Conecte esta aplicación con los resultados de laboratorio para ayudar a diagnosticar el cáncer de mama a partir de una muestra de tejido. Esta aplicación predice mediante un modelo de inteligencia artificial si una masa mamaria es benigna o maligna en función de las mediciones que recibe del médico.")
    col1, col2 = st.columns([4,1])
    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
    with col2:
        add_predictions(input_data)
    
    if st.button('Visualizar Análisis Avanzados'):
        st.subheader("Árbol de Decisión")
        visualize_decision_tree(input_data)
        st.subheader("KMeans Clustering")
        visualize_kmeans()

if __name__ == '__main__':
    main()