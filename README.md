# Análisis de Potencia Fotovoltaica: Caso I.E. Nuestra Señora de la Candelaria

Este proyecto desarrolla un **Análisis Correlacional, Modelamiento Físico y Predictivo Avanzado** para evaluar la precisión de la estimación de potencia en un sistema fotovoltaico on-grid ubicado en Malambo, Atlántico. 

El núcleo de la investigación compara los datos de generación real del inversor frente a las estimaciones teóricas basadas en datos climáticos de la estación meteorológica del Aeropuerto Ernesto Cortissoz (ubicada a ~5 km del sitio de estudio), utilizando **Machine Learning (CatBoost)** para cerrar la brecha de precisión causada por la varianza microclimática local y las dinámicas transitorias de nubosidad tropical.

---

## 📊 Resumen del Pipeline Metodológico

El flujo de trabajo implementado en el ecosistema de Python se divide en tres fases analíticas:

1. **Modelamiento Físico de Referencia:** Simulación de la potencia teórica del sistema (4 paneles solares de 340W) mediante el modelo físico de un diodo y estimaciones de irradiancia proxy (`pvlib`).
2. **Análisis de Residuos y Brecha de Microclima:** Evaluación espacial y temporal de la subestimación sistemática del modelo físico frente a la realidad local. Se identifican las brechas asociadas a efectos locales de nubosidad y fallos operativos (caídas por desconexión de red o efecto isla).
3. **Machine Learning No Lineal:** Implementación y entrenamiento de modelos no lineales (**Random Forest** y **CatBoost Regressor**) para capturar las complejas interacciones meteorológicas locales, superando con creces la regresión lineal multivariada (OLS).

---

## 📈 Métricas de Rendimiento y Hallazgos Clave

* **Salto Predictivo Radical:** La regresión lineal (OLS) es incapaz de modelar la complejidad del sistema ($R^2 = 0.08$ y un alto MAE de 242.06 W). El modelo no lineal **CatBoost** incrementó drásticamente la capacidad explicativa, logrando un **$R^2 = 0.58$** en el set de prueba.
* **Reducción de Errores de Predicción:** El Error Absoluto Medio (MAE) se redujo sistemáticamente conforme se incrementó la sofisticación del modelado:
  * **Modelo Físico (Diodo):** $MAE = 173.06 \text{ W}$ (Desviación constante por distancia de la estación).
  * **Regresión Lineal (OLS):** $MAE = 242.06 \text{ W}$ (Fracaso por suponer linealidad en factores atmosféricos).
  * **Random Forest:** $MAE = 125.79 \text{ W}$ ($R^2 = 0.556$).
  * **CatBoost Regressor:** **$MAE = 123.57 \text{ W}$** ($R^2 = 0.580$ - Modelo óptimo y definitivo).
* **Importancia de Variables (Feature Importance):** La **Irradiancia Proxy** representa el factor preponderante en el entrenamiento de los árboles, seguida por la **Temperatura Ambiente** y el **Índice UV** (factores críticos que alteran la curva de eficiencia del silicio de los paneles).

---

## 🛠️ Tecnologías y Librerías Utilizadas

* **Entorno:** Python 3.13
* **Librerías Clave:** 
  * `CatBoost` para el modelado de gradiente boosting no lineal (optimizado en memoria mediante `allow_writing_files=False`).
  * `Scikit-Learn` para ensambles de Random Forest, división de datos y métricas de validación.
  * `pvlib` para la simulación física de la curva del panel solar.
  * `statsmodels` para diagnósticos estadísticos de heterocedasticidad (Breusch-Pagan) y OLS.
  * `pandas` & `numpy` para procesamiento, limpieza estructural de nulos e imputación.
  * `seaborn` & `matplotlib` para la generación y exportación de visuales vectoriales.

---

## 📊 Catálogo de Visuales Académicos (`visuals/`)

El repositorio contiene exactamente **13 figuras académicas secuenciales** exportadas en alta resolución (300 DPI) para su inclusión directa en artículos científicos, posters o presentaciones ejecutivas:

* **`fig1_estructura_datos.png`**: Mapa de calor de valores nulos para el control de calidad estructural del dataset.
* **`fig2_serie_temporal_potencia.png`**: Historial continuo de generación real capturado por el inversor Hoymiles.
* **`fig3_comportamiento_irradiancia.png`**: Historial temporal de la irradiancia estimada proxy calculada en el sitio.
* **`fig4_analisis_irradiancia.png`**: Diagrama de cajas de 6 franjas diurnas estructuradas (desde Mañana Temprana hasta Tarde Final) que muestra la distribución de energía.
* **`fig5_potencia_vs_irradiancia.png`**: Gráfico de dispersión empírico que ilustra la relación no lineal entre la potencia real y la irradiancia proxy.
* **`fig6_multidia_real_teorica.png`**: Comparativa multi-día (6 paneles de fecha) que resalta en color **verde suave translúcido** la brecha de potencia causada por dinámicas microclimáticas locales.
* **`fig7_histograma_residuos_fisico.png`**: Histograma de distribución de residuos del modelo físico que evidencia una subestimación sistemática.
* **`fig8_scatter_correlacion.png`**: Gráfico de dispersión con línea de ajuste para evaluar la correlación real vs estimación teórica.
* **`fig9_heterocedasticidad_residuos.png`**: Diagnóstico de varianza residual vs valores ajustados, confirmando heterocedasticidad en el modelo OLS lineal.
* **`fig10_histograma_residuos_ols.png`**: Histograma de errores de predicción de la regresión lineal multivariada.
* **`fig11_real_vs_predicho_ml.png`**: Gráfica de dispersión comparativa: Potencia Real vs Predicciones de Machine Learning (Random Forest vs CatBoost).
* **`fig12_comparativa_mae.png`**: Gráfico de barras comparativo del Error Absoluto Medio (MAE) en vatios para los modelos analizados.
* **`fig13_serie_temporal_completa.png`**: Comparativa de series temporales continuas a lo largo de 5 días para validar la consistencia predictiva de los modelos.

---

## 📁 Estructura del Repositorio

* **`Jupyter/`**: Contiene el notebook de Jupyter principal `Caso_Estudio_Solar_Malambo.ipynb`, completamente ejecutado, limpio de códigos huérfanos y guardado con sus salidas inline.
* **`visuals/`**: Carpeta de exportación oficial que alberga las 13 figuras numeradas secuencialmente en formato `.png`.
* **`csv/`**: Archivos de datos unificados en formato Excel (`ANALISIS_SOLAR_UNIFICADO.xlsx`).
* **`docs/`**: Documentos adicionales y PDF de la presentación final del caso de estudio.
* **`requirements.txt`**: Archivo de requisitos que describe la versión exacta de las dependencias necesarias para garantizar la reproducibilidad científica total.

---
**Autor:** Maria Jose Leal Brochero  
*Estudiante de la Maestría en Gestión Energética - Universidad del Atlántico*