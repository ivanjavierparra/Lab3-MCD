import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Configuraciones de estilo
st.set_page_config(layout="wide")
sns.set(style="whitegrid")

# === 1. CARGA DE DATOS ===
@st.cache_data
def load_data():
    df = pd.read_csv("datasets/sell-in.csv", sep='\t')
    productos = pd.read_csv("datasets/tb_productos.csv", sep='\t')

    df = df.merge(productos, on="product_id", how="left")

    # Parsear periodo
    df['periodo'] = pd.to_datetime(df['periodo'].astype(str), format='%Y%m')
    df['anio_mes'] = df['periodo'].dt.to_period('M').dt.to_timestamp()
    df['anio_mes'] = df['anio_mes'].astype(str)
    
    # Agrupación
    df_grouped = df.groupby(['anio_mes', 'cat1', 'cat2', 'cat3'])['tn'].sum().reset_index()

    # Categoría combinada
    df_grouped['categoria_completa'] = (
        df_grouped['cat1'].astype(str) + " / " +
        df_grouped['cat2'].astype(str) + " / " +
        df_grouped['cat3'].astype(str)
    )

    return df_grouped

df = load_data()

# === 2. SIDEBAR: FILTROS ===
st.sidebar.header("Filtros de categoría")

top_categorias = (
    df.groupby('categoria_completa')['tn']
    .sum()
    .sort_values(ascending=False)
    .head(15)
    .index
)

modo = st.sidebar.radio("Modo de selección", ["Top 15", "Todas"])
if modo == "Top 15":
    opciones = top_categorias
else:
    opciones = df['categoria_completa'].unique()

categorias_seleccionadas = st.sidebar.multiselect(
    "Seleccioná categorías",
    options=opciones,
    default=opciones[:3] if len(opciones) > 3 else opciones
)

df_filtrado = df[df['categoria_completa'].isin(categorias_seleccionadas)]

# === 3. TÍTULO Y DESCRIPCIÓN ===
st.title("📊 Evolución de Ventas por Categorías (cat1 / cat2 / cat3)")
st.markdown("""
Esta herramienta permite visualizar la evolución de ventas mensuales (en toneladas) para combinaciones específicas de categorías.  
Podés filtrar las categorías en el panel lateral.
""")

# === 4. GRÁFICO ===
if df_filtrado.empty:
    st.warning("No hay datos para mostrar con los filtros seleccionados.")
else:
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.lineplot(data=df_filtrado, x='anio_mes', y='tn', hue='categoria_completa', marker='o', ax=ax)
    ax.set_title("Evolución de Ventas por Categoría")
    ax.set_xlabel("Periodo")
    ax.set_ylabel("Toneladas Vendidas")
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title="Categoría", bbox_to_anchor=(1.02, 1), loc='upper left')
    st.pyplot(fig)
