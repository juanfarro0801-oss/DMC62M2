import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

# ==========================================
# CLASE DE PROGRAMACIÓN ORIENTADA A OBJETOS
# ==========================================
class DataAnalyzer:
    """Clase para encapsular el análisis exploratorio de datos."""
    def __init__(self, df):
        self.df = df
        self.num_cols, self.cat_cols = self.classify_variables()

    def classify_variables(self):
        """Clasifica las variables en numéricas y categóricas."""
        num_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = self.df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        return num_cols, cat_cols

    def get_descriptive_stats(self):
        """Retorna las estadísticas descriptivas del dataset."""
        return self.df.describe()
    
    def get_missing_values(self):
        """Retorna el conteo de valores nulos."""
        return self.df.isnull().sum()

    def plot_histogram(self, column, bins):
        """Genera un histograma para una variable numérica."""
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(self.df[column], bins=bins, kde=True, ax=ax, color='skyblue')
        ax.set_title(f'Distribución de {column}')
        return fig

    def plot_bar(self, column):
        """Genera un gráfico de barras para una variable categórica."""
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(data=self.df, x=column, order=self.df[column].value_counts().index, palette='viridis', ax=ax)
        plt.xticks(rotation=45)
        ax.set_title(f'Frecuencia de {column}')
        return fig

    def plot_bivariate_num_cat(self, num_col, cat_col):
        """Genera un boxplot para comparar una variable numérica vs categórica."""
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(data=self.df, x=cat_col, y=num_col, palette='Set2', ax=ax)
        plt.xticks(rotation=45)
        ax.set_title(f'{num_col} vs {cat_col}')
        return fig

    def plot_bivariate_cat_cat(self, cat_col1, cat_col2):
        """Genera un gráfico de barras apiladas o agrupadas para dos categóricas."""
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(data=self.df, x=cat_col1, hue=cat_col2, palette='coolwarm', ax=ax)
        plt.xticks(rotation=45)
        ax.set_title(f'{cat_col1} vs {cat_col2}')
        return fig

# ==========================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================
st.set_page_config(page_title="EDA - Bank Marketing", layout="wide", page_icon="🏦")

# ==========================================
# SIDEBAR - NAVEGACIÓN MENÚ PRINCIPAL
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=100)
st.sidebar.title("Navegación")

# Solo dos módulos en el menú lateral
menu = st.sidebar.radio("Seleccione un Módulo:", 
                        ["🏠 Home", 
                         "📂 Carga y Análisis (EDA)"])

# ==========================================
# MÓDULO 1: HOME
# ==========================================
if menu == "🏠 Home":
    st.title("🏦 Proyecto Aplicado: Bank Marketing EDA")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Objetivo del Análisis")
        st.write("""
        El objetivo principal de este proyecto es realizar un Análisis Exploratorio de Datos (EDA)
        interactivo para comprender los factores demográficos y financieros que influyen en que 
        un cliente acepte o rechace una campaña de marketing de depósitos a plazo.
        """)
        st.subheader("El Problema")
        st.write("""
        Durante los últimos 6 meses, la efectividad de las campañas comerciales del banco cayó de 12% a 8%. 
        A través de este análisis buscaremos descubrir relaciones y comportamientos relevantes para revertir esta tendencia.
        """)
        
    with col2:
        st.subheader("Datos del Autor")
        st.write("**👤 Nombre:** JUAN DIEGO FARRO TAZA")
        st.write("**📚 Curso:** Especialización en Python for Analytics")
        st.write("**📅 Año:** 2026")
        
        st.subheader("Tecnologías Utilizadas")
        st.write("- Python 🐍")
        st.write("- Pandas y NumPy 🐼")
        st.write("- Matplotlib y Seaborn 📊")
        st.write("- Streamlit 🚀")

# ==========================================
# MÓDULO 2: CARGA DEL DATASET Y EDA INTEGRADO
# ==========================================
elif menu == "📂 Carga y Análisis (EDA)":
    st.title("📂 Carga de Datos y EDA")
    st.write("Sube el archivo `BankMarketing.csv` para desplegar automáticamente el Análisis Exploratorio.")
    
    uploaded_file = st.file_uploader("Selecciona el archivo CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Lectura del dataset
            df = pd.read_csv(uploaded_file, sep=";")
            if len(df.columns) == 1:
                df = pd.read_csv(uploaded_file, sep=",")
                
            st.success("✅ Archivo cargado correctamente.")
            
            # Vista previa colapsable para ahorrar espacio visual
            with st.expander("Ver vista previa y dimensiones del dataset", expanded=False):
                st.info(f"El dataset contiene **{df.shape[0]} filas** y **{df.shape[1]} columnas**.")
                st.dataframe(df.head())
            
            st.markdown("---")
            st.title("📊 Análisis Exploratorio de Datos (EDA)")
            
            # Instanciar clase POO
            analyzer = DataAnalyzer(df)
            
            # Uso de TABS: Agregamos la pestaña 11 para las conclusiones
            tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
                "1. Info", "2. Clasificación", "3. Estadísticas", "4. Nulos", 
                "5. Dist. Numéricas", "6. Dist. Categóricas", "7. Bivariado (Num vs Cat)", 
                "8. Bivariado (Cat vs Cat)", "9. Filtros Dinámicos", "10. Hallazgos", "11. Conclusiones"
            ])

            # Ítem 1: Información general
            with tab1:
                st.header("Ítem 1: Información General del Dataset")
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Tipos de Datos")
                    st.dataframe(df.dtypes.rename("Tipo de Dato").astype(str))
                with col2:
                    st.subheader("Resumen (.info)")
                    buffer = io.StringIO()
                    df.info(buf=buffer)
                    s = buffer.getvalue()
                    st.text(s)

            # Ítem 2: Clasificación de variables
            with tab2:
                st.header("Ítem 2: Clasificación de Variables")
                st.write("Identificación usando la clase personalizada `DataAnalyzer`.")
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"**Variables Numéricas ({len(analyzer.num_cols)}):**")
                    for col in analyzer.num_cols:
                        st.write(f"- {col}")
                with col2:
                    st.info(f"**Variables Categóricas ({len(analyzer.cat_cols)}):**")
                    for col in analyzer.cat_cols:
                        st.write(f"- {col}")

            # Ítem 3: Estadísticas descriptivas
            with tab3:
                st.header("Ítem 3: Estadísticas Descriptivas")
                st.dataframe(analyzer.get_descriptive_stats())
                st.markdown("""
                **Interpretación Básica:**
                - Se utiliza `mean` (media) y `50%` (mediana) para evaluar el centro de los datos. 
                - Si hay una gran diferencia entre media y mediana (ej. `duration`), indica que la distribución está sesgada (asimetría).
                - `std` (Desviación estándar) indica la dispersión de los datos respecto a la media.
                """)

            # Ítem 4: Valores faltantes
            with tab4:
                st.header("Ítem 4: Análisis de Valores Faltantes")
                nulls = analyzer.get_missing_values()
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Conteo de Nulos por Columna:**")
                    if nulls.sum() > 0:
                        st.dataframe(nulls[nulls > 0])
                    else:
                        st.success("No se encontraron valores nulos.")
                with col2:
                    if nulls.sum() > 0:
                        fig, ax = plt.subplots()
                        sns.heatmap(df.isnull(), cbar=False, cmap='viridis', ax=ax)
                        st.pyplot(fig)
                    else:
                        st.success("El dataset está limpio, no requiere imputación de datos nulos en primera instancia.")

            # Ítem 5: Distribución de variables numéricas
            with tab5:
                st.header("Ítem 5: Distribución Numérica")
                num_sel = st.selectbox("Seleccione la variable numérica:", analyzer.num_cols, key="num_hist")
                bins = st.slider("Número de Bins para el Histograma", min_value=10, max_value=100, value=30)
                if num_sel:
                    st.pyplot(analyzer.plot_histogram(num_sel, bins))
                    st.write(f"Interpretación: Observamos la concentración de los clientes respecto a su **{num_sel}**.")

            # Ítem 6: Análisis de variables categóricas
            with tab6:
                st.header("Ítem 6: Análisis de Variables Categóricas")
                cat_sel = st.selectbox("Seleccione la variable categórica:", analyzer.cat_cols, key="cat_bar")
                if cat_sel:
                    st.pyplot(analyzer.plot_bar(cat_sel))
                    proporciones = df[cat_sel].value_counts(normalize=True) * 100
                    st.write("**Proporciones (%)**")
                    st.dataframe(proporciones.round(2))

            # Ítem 7: Análisis Bivariado (Numérico vs Categórico)
            with tab7:
                st.header("Ítem 7: Numérico vs Categórico")
                col_num = st.selectbox("Eje Y (Numérica):", analyzer.num_cols, index=analyzer.num_cols.index('age') if 'age' in analyzer.num_cols else 0)
                col_cat = st.selectbox("Eje X (Categórica - ej. Target 'y'):", analyzer.cat_cols, index=analyzer.cat_cols.index('y') if 'y' in analyzer.cat_cols else 0)
                st.pyplot(analyzer.plot_bivariate_num_cat(col_num, col_cat))
                st.write(f"Evalúa cómo varía la distribución de **{col_num}** dependiendo de la categoría en **{col_cat}**.")

            # Ítem 8: Análisis Bivariado (Categórico vs Categórico)
            with tab8:
                st.header("Ítem 8: Categórico vs Categórico")
                cat_1 = st.selectbox("Variable Principal (Eje X):", analyzer.cat_cols, index=analyzer.cat_cols.index('job') if 'job' in analyzer.cat_cols else 0)
                cat_2 = st.selectbox("Variable de Agrupación (Color - ej. 'y'):", analyzer.cat_cols, index=analyzer.cat_cols.index('y') if 'y' in analyzer.cat_cols else 0)
                st.pyplot(analyzer.plot_bivariate_cat_cat(cat_1, cat_2))

            # Ítem 9: Análisis dinámico / Filtros
            with tab9:
                st.header("Ítem 9: Análisis Dinámico Basado en Parámetros")
                st.write("Usa los widgets para filtrar y visualizar el dataset dinámicamente.")
                selected_cols = st.multiselect("Seleccione las columnas a visualizar:", df.columns.tolist(), default=['age', 'job', 'marital', 'y'])
                
                if 'age' in df.columns:
                    age_range = st.slider("Filtro por Edad", int(df['age'].min()), int(df['age'].max()), (30, 50))
                else:
                    age_range = None
                    
                solo_exitosos = st.checkbox("Mostrar solo campañas exitosas (y = 'yes')")
                
                filtered_df = df.copy()
                if age_range:
                    filtered_df = filtered_df[(filtered_df['age'] >= age_range[0]) & (filtered_df['age'] <= age_range[1])]
                if solo_exitosos and 'y' in df.columns:
                    filtered_df = filtered_df[filtered_df['y'] == 'yes']
                    
                st.write(f"Mostrando {len(filtered_df)} registros después de aplicar los filtros.")
                st.dataframe(filtered_df[selected_cols].head(100))

            # Ítem 10: Hallazgos clave
            with tab10:
                st.header("Ítem 10: Hallazgos Clave")
                st.write("Visualización Resumen - Matriz de Correlación")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(df[analyzer.num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
                st.pyplot(fig)
                st.markdown("""
                ### 💡 Insights Principales:
                1. **Impacto de la Duración**: La variable `duration` suele tener la mayor correlación con el éxito de la campaña. Sin embargo, no es un buen predictor para modelos futuros porque solo se conoce al finalizar la llamada.
                2. **Factores Macroeconómicos**: Las tasas de empleo (`emp.var.rate`) y el Euribor (`euribor3m`) influyen en el comportamiento conservador del cliente ante los depósitos.
                3. **Perfil Demográfico**: Ciertos trabajos (ej. jubilados o estudiantes) pueden mostrar tasas de aceptación proporcionalmente mayores.
                """)

            # Ítem 11: Conclusiones (Nueva Pestaña)
            with tab11:
                st.header("Ítem 11: Conclusiones para la Toma de Decisiones")
                st.markdown("""
                En base al Análisis Exploratorio de Datos realizado sobre la caída de efectividad de las campañas comerciales (del 12% al 8%), se extraen las siguientes **5 conclusiones orientadas al negocio**:

                1. **Redirección de Esfuerzos por Perfil (Targeting):** 
                   Se observa que grupos poblacionales específicos (por ejemplo, personas en etapa de jubilación o con ciertos perfiles profesionales) tienen una tasa de conversión superior. El equipo de ventas debe priorizar estas bolsas de clientes en lugar de realizar llamadas masivas aleatorias.
                   
                2. **Optimización del Canal y Tiempo de Contacto:** 
                   Los datos sugieren que meses específicos o días de la semana rinden mejor. Se debe reprogramar el calendario de los agentes comerciales para intensificar los contactos durante los picos históricos de mayor aceptación y reducir esfuerzo en meses de baja conversión.
                   
                3. **La Duración del Contacto como Indicador de Calidad:** 
                   Las llamadas exitosas muestran una duración media sustancialmente mayor. Se debe capacitar a los ejecutivos comerciales en habilidades blandas y "rompehielos" que logren retener al cliente en la línea los primeros 60 segundos vitales, en lugar de intentar forzar un cierre rápido.
                   
                4. **Adaptación a las Condiciones Macroeconómicas:** 
                   Variables como la tasa de empleo o el índice de precios muestran una influencia clara. El banco debe ajustar su guion de ventas (pitch comercial) de los productos a plazo fijo, destacándolos como un "refugio seguro" ante momentos de alta incertidumbre económica.
                   
                5. **Control de la Frecuencia (Fatiga del Cliente):** 
                   La variable de contactos durante la misma campaña (`campaign`) indica que insistir repetidas veces a un mismo cliente genera rendimientos decrecientes y potencial rechazo. Se debe establecer una política estricta de máximo de contactos por campaña para evitar la fatiga del cliente y optimizar el tiempo del agente comercial.
                """)
                st.info("📌 **Nota:** Este dashboard fue diseñado aplicando principios de limpieza visual, modularidad (POO) y componentes dinámicos requeridos para el portafolio profesional.")

        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")
    else:
        st.info("👆 Esperando la carga del archivo CSV para desplegar el análisis estadístico.")
