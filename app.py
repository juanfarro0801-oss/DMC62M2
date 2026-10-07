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
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=self.df, x=cat_col, y=num_col, palette='Set2', ax=ax)
        plt.xticks(rotation=45)
        ax.set_title(f'{num_col} vs {cat_col}')
        return fig

    def plot_bivariate_cat_cat(self, cat_col1, cat_col2):
        """Genera un gráfico de barras agrupadas para dos categóricas."""
        fig, ax = plt.subplots(figsize=(12, 6))
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
    st.write("Sube el archivo `BankMarketing.csv` para desplegar el Análisis Exploratorio.")
    
    uploaded_file = st.file_uploader("Selecciona el archivo CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Lectura del dataset
            df = pd.read_csv(uploaded_file, sep=";")
            if len(df.columns) == 1:
                df = pd.read_csv(uploaded_file, sep=",")
                
            st.success("✅ Archivo cargado correctamente.")
            
            with st.expander("Ver vista previa y dimensiones del dataset", expanded=False):
                st.info(f"El dataset contiene **{df.shape[0]} filas** y **{df.shape[1]} columnas**.")
                st.dataframe(df.head())
            
            st.markdown("---")
            st.title("📊 Análisis Exploratorio de Datos (EDA)")
            
            analyzer = DataAnalyzer(df)
            
            tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
                "1. Info", "2. Clasif.", "3. Estadísticas", "4. Nulos", 
                "5. Dist. Num", "6. Dist. Cat", "7. Num vs Cat", 
                "8. Cat vs Cat", "9. Filtros", "10. Correlación", "11. Conclusiones"
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
                    st.text(buffer.getvalue())

            # Ítem 2: Clasificación de variables
            with tab2:
                st.header("Ítem 2: Clasificación de Variables")
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"**Variables Numéricas ({len(analyzer.num_cols)}):**")
                    st.write(", ".join(analyzer.num_cols))
                with col2:
                    st.info(f"**Variables Categóricas ({len(analyzer.cat_cols)}):**")
                    st.write(", ".join(analyzer.cat_cols))

            # Ítem 3: Estadísticas descriptivas
            with tab3:
                st.header("Ítem 3: Estadísticas Descriptivas")
                st.dataframe(analyzer.get_descriptive_stats())
                st.info("""
                **Interpretación de la Tabla:**
                - **Edad (`age`):** La edad media de los clientes contactados es de 40 años, con un mínimo de 17 y un máximo de 98 años.
                - **Duración (`duration`):** Presenta una asimetría fuerte. El 75% de las llamadas dura 319 segundos o menos, pero el valor máximo alcanza los 4918 segundos, indicando la presencia de valores atípicos (llamadas inusualmente largas).
                - **Contactos (`campaign`):** En promedio se han realizado 2.5 contactos por cliente durante esta campaña.
                """)

            # Ítem 4: Valores faltantes
            with tab4:
                st.header("Ítem 4: Análisis de Valores Faltantes")
                nulls = analyzer.get_missing_values()
                if nulls.sum() > 0:
                    st.dataframe(nulls[nulls > 0])
                else:
                    st.success("El dataset está limpio, no se encontraron valores nulos (missing values).")

            # Ítem 5: Distribución de variables numéricas
            with tab5:
                st.header("Ítem 5: Distribución Numérica")
                num_sel = st.selectbox("Seleccione la variable numérica:", analyzer.num_cols, index=0)
                bins = st.slider("Número de Bins", min_value=10, max_value=100, value=30)
                st.pyplot(analyzer.plot_histogram(num_sel, bins))
                
                if num_sel == 'age':
                    st.warning("""
                    **Interpretación del Gráfico (Distribución de age):**
                    La distribución tiene forma de campana pero con un claro sesgo hacia la derecha (cola larga). La gran mayoría de los clientes contactados se concentran entre los **30 y 40 años**. El volumen de llamadas disminuye drásticamente a partir de los 60 años.
                    """)

            # Ítem 6: Análisis de variables categóricas
            with tab6:
                st.header("Ítem 6: Análisis de Variables Categóricas")
                cat_sel = st.selectbox("Seleccione la variable categórica:", analyzer.cat_cols, index=analyzer.cat_cols.index('job') if 'job' in analyzer.cat_cols else 0)
                st.pyplot(analyzer.plot_bar(cat_sel))
                
                if cat_sel == 'job':
                    st.warning("""
                    **Interpretación del Gráfico (Frecuencia de job):**
                    Las profesiones más contactadas por el banco son los perfiles **administrativos (`admin.`)**, seguidos de los trabajadores manuales (`blue-collar`) y los técnicos (`technician`). En contraste, los estudiantes y desempleados representan la porción más pequeña de la base de datos.
                    """)

            # Ítem 7: Análisis Bivariado (Numérico vs Categórico)
            with tab7:
                st.header("Ítem 7: Numérico vs Categórico")
                col_num = st.selectbox("Eje Y (Numérica):", analyzer.num_cols, index=analyzer.num_cols.index('age') if 'age' in analyzer.num_cols else 0)
                col_cat = st.selectbox("Eje X (Categórica):", analyzer.cat_cols, index=analyzer.cat_cols.index('y') if 'y' in analyzer.cat_cols else 0)
                st.pyplot(analyzer.plot_bivariate_num_cat(col_num, col_cat))
                
                if col_num == 'age' and col_cat == 'y':
                    st.warning("""
                    **Interpretación del Gráfico (age vs y):**
                    Las cajas (cuartiles) de las personas que aceptaron (`yes`) y rechazaron (`no`) la oferta son bastante similares en su tendencia central. Sin embargo, se observa una notable concentración de **valores atípicos (outliers) en edades avanzadas (70 a casi 100 años)** dentro del grupo que sí aceptó el depósito (`yes`), sugiriendo que la tercera edad podría tener una mayor predisposición a la conversión.
                    """)

            # Ítem 8: Análisis Bivariado (Categórico vs Categórico)
            with tab8:
                st.header("Ítem 8: Categórico vs Categórico")
                cat_1 = st.selectbox("Eje X:", analyzer.cat_cols, index=analyzer.cat_cols.index('job') if 'job' in analyzer.cat_cols else 0)
                cat_2 = st.selectbox("Color (hue):", analyzer.cat_cols, index=analyzer.cat_cols.index('y') if 'y' in analyzer.cat_cols else 0)
                st.pyplot(analyzer.plot_bivariate_cat_cat(cat_1, cat_2))
                
                if cat_1 == 'job' and cat_2 == 'y':
                    st.warning("""
                    **Interpretación del Gráfico (job vs y):**
                    Aunque la categoría `admin.` tiene el mayor volumen absoluto de aceptaciones (`yes`), las barras reflejan que grupos como **los jubilados (`retired`) y los estudiantes (`student`)** tienen una proporción de éxito mucho mejor comparada con sus rechazos (`no`). Por otro lado, los trabajadores `blue-collar` presentan una de las tasas de rechazo proporcionalmente más altas.
                    """)

            # Ítem 9: Análisis dinámico / Filtros
            with tab9:
                st.header("Ítem 9: Filtros Dinámicos")
                selected_cols = st.multiselect("Columnas a visualizar:", df.columns.tolist(), default=['age', 'job', 'marital', 'y'])
                if 'age' in df.columns:
                    age_range = st.slider("Filtro Edad", int(df['age'].min()), int(df['age'].max()), (30, 50))
                solo_exitosos = st.checkbox("Solo exitosos (y='yes')")
                
                filtered_df = df.copy()
                if 'age' in df.columns:
                    filtered_df = filtered_df[(filtered_df['age'] >= age_range[0]) & (filtered_df['age'] <= age_range[1])]
                if solo_exitosos and 'y' in df.columns:
                    filtered_df = filtered_df[filtered_df['y'] == 'yes']
                    
                st.write(f"Mostrando {len(filtered_df)} registros.")
                st.dataframe(filtered_df[selected_cols].head(100))

            # Ítem 10: Matriz de Correlación
            with tab10:
                st.header("Ítem 10: Matriz de Correlación")
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.heatmap(df[analyzer.num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
                st.pyplot(fig)
                st.warning("""
                **Interpretación del Gráfico (Correlaciones):**
                Se evidencia una **fuerte multicolinealidad** entre las variables macroeconómicas. El indicador de empleo (`emp.var.rate`), la tasa Euribor (`euribor3m`) y el número de empleados (`nr.employed`) presentan correlaciones positivas altísimas entre sí (superiores a 0.90). Esto indica que el contexto económico global del país en el momento de la llamada se mueve en bloque e influye directamente en el comportamiento financiero del cliente.
                """)

            # Ítem 11: Conclusiones
            with tab11:
                st.header("Ítem 11: Conclusiones para la Toma de Decisiones")
                st.markdown("""
                Con base en las interpretaciones de los datos visualizados para abordar la caída de la efectividad comercial, se presentan las siguientes conclusiones estratégicas:

                1. **Micro-Segmentación Rentable (Targeting Demográfico):**
                   Los datos demuestran que el banco gasta muchos recursos llamando a trabajadores manuales (`blue-collar`), quienes tienen un volumen alto de rechazo. Por otro lado, las personas de la tercera edad (70+ años) y jubilados (`retired`), así como estudiantes, tienen proporciones de aceptación mucho más altas. **Acción:** Redirigir el esfuerzo de llamadas masivas hacia nichos específicos como jubilados que buscan seguridad financiera.

                2. **Calidad de la Llamada sobre la Cantidad:**
                   La enorme dispersión en la duración de la llamada (hasta 4918 segundos) indica que las interacciones exitosas requieren retener al usuario. **Acción:** Capacitar a los asesores para no forzar cierres rápidos en el primer minuto, sino entablar una conversación consultiva, ya que una mayor duración está ligada al éxito.

                3. **Impacto del Entorno Macroeconómico:**
                   La altísima correlación entre el Euribor, las tasas de variación de empleo y los índices de precios (>0.90) confirma que el cliente reacciona en bloque al contexto económico. **Acción:** Adaptar el discurso de venta dinámicamente; si el Euribor está a la baja, el depósito a plazo debe venderse como un "refugio preventivo" antes de que las tasas caigan más.

                4. **El Desafío de la Edad Central:**
                   El grueso de las llamadas (la gran masa entre 30 y 40 años) coincide con la base laboral activa (`admin.`, `technician`). Aunque aportan en volumen absoluto, su tasa de conversión está estancada. **Acción:** Para este grupo, el producto clásico de depósito no es atractivo. Se deben diseñar campañas cruzadas ofreciendo flexibilidades o tasas diferenciadas para recuperar el porcentaje perdido en este segmento poblacional.
                """)

        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")
    else:
        st.info("👆 Esperando la carga del archivo CSV para desplegar el análisis.")
