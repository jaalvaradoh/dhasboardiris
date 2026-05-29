import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------

st.set_page_config(
    page_title="Iris Dashboard",
    page_icon="🌸",
    layout="wide"
)

# ---------------------------------------------------
# TÍTULO
# ---------------------------------------------------

st.title("🌸 Iris Analytics Dashboard")
st.caption("Dashboard interactivo profesional con Streamlit y Machine Learning")

# ---------------------------------------------------
# CARGA DE DATOS
# ---------------------------------------------------

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["species"] = [iris.target_names[i] for i in iris.target]

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.header("⚙️ Configuración")

species_selected = st.sidebar.multiselect(
    "Selecciona especies",
    options=df["species"].unique(),
    default=df["species"].unique()
)

filtered_df = df[df["species"].isin(species_selected)]

# ---------------------------------------------------
# KPIs
# ---------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total registros", len(filtered_df))
col2.metric("Variables", 4)
col3.metric("Especies", filtered_df["species"].nunique())
col4.metric(
    "Sepal Length promedio",
    round(filtered_df["sepal length (cm)"].mean(), 2)
)

st.divider()

# ---------------------------------------------------
# GRÁFICOS PRINCIPALES
# ---------------------------------------------------

left, right = st.columns(2)

with left:

    st.subheader("Distribución de especies")

    fig_bar = px.histogram(
        filtered_df,
        x="species",
        color="species",
        template="plotly",
        text_auto=True
    )

    fig_bar.update_layout(
        showlegend=False,
        height=450
    )

    st.plotly_chart(fig_bar, use_container_width=True)

with right:

    st.subheader("Relación entre variables")

    fig_scatter = px.scatter(
        filtered_df,
        x="sepal length (cm)",
        y="petal length (cm)",
        color="species",
        size="petal width (cm)",
        hover_data=filtered_df.columns,
        template="plotly",
        height=450
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

# ---------------------------------------------------
# MATRIZ DE CORRELACIÓN
# ---------------------------------------------------

st.subheader("Mapa de correlación")

corr = filtered_df.drop(columns=["species"]).corr()

fig_corr = px.imshow(
    corr,
    text_auto=True,
    color_continuous_scale="Blues",
    aspect="auto"
)

fig_corr.update_layout(height=500)

st.plotly_chart(fig_corr, use_container_width=True)

# ---------------------------------------------------
# MACHINE LEARNING
# ---------------------------------------------------

st.subheader("Modelo Predictivo")

X = df.drop(columns=["species"])
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

colA, colB = st.columns([1, 2])

with colA:

    st.metric(
        "Accuracy",
        f"{accuracy:.2%}"
    )

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    })

    fig_importance = px.bar(
        importance_df.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="viridis",
        template="plotly"
    )

    st.plotly_chart(fig_importance, use_container_width=True)

with colB:

    cm = confusion_matrix(y_test, predictions)

    fig_cm = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=iris.target_names,
            y=iris.target_names,
            text=cm,
            texttemplate="%{text}",
            colorscale="Blues"
        )
    )

    fig_cm.update_layout(
        title="Matriz de Confusión",
        height=500
    )

    st.plotly_chart(fig_cm, use_container_width=True)

# ---------------------------------------------------
# DATASET
# ---------------------------------------------------

st.subheader("Vista del Dataset")

st.dataframe(
    filtered_df,
    use_container_width=True,
    hide_index=True
)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.divider()

st.caption(
    "Desarrollado con Streamlit • Plotly • Scikit-Learn"
)
