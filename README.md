# Análisis de Potencia Fotovoltaica: Caso I.E. Nuestra Señora de la Candelaria

Este proyecto desarrolla un **Análisis Correlacional y Modelado Predictivo Avanzado** para evaluar la precisión de la estimación de potencia en un sistema fotovoltaico on-grid ubicado en Malambo, Atlántico. 

El núcleo de la investigación compara los datos de generación real del inversor frente a las estimaciones teóricas basadas en datos climáticos del Aeropuerto Ernesto Cortissoz, utilizando **Machine Learning** para cerrar la brecha de precisión causada por la varianza microclimática local.

## 📊 Resumen del Análisis
Se implementó un pipeline de datos en Python para:
1. **Modelamiento Físico:** Cálculo de la generación teórica mediante el modelo de un diodo (PVLib).
2. **Análisis de Residuos:** Identificación de brechas de potencia entre el modelo regional y la realidad local.
3. **Machine Learning No Lineal:** Implementación de modelos de **Random Forest** y **XGBoost** para predecir la potencia basándose en múltiples variables meteorológicas, superando las limitaciones de la regresión lineal.

## 🛠️ Tecnologías Utilizadas
* **Lenguaje:** Python 3.13
* **Librerías principales:** 
    * `XGBoost` & `Scikit-Learn` para modelado predictivo avanzado.
    * `pvlib` para simulación física de sistemas solares.
    * `pandas` & `numpy` para procesamiento de datos.
    * `statsmodels` para la regresión multivariada inicial.
    * `seaborn` & `matplotlib` para visualización de alta resolución.

## 📈 Hallazgos Clave
* **Mejora Predictiva:** El uso de modelos no lineales (**XGBoost**) incrementó la capacidad de explicación del modelo de un **$R^2 = 0.08$** (modelo lineal) a un **$R^2 > 0.52$**, demostrando que el impacto meteorológico es altamente complejo y no lineal.
* **Importancia de Variables:** La **Irradiancia Proxy** es el factor dominante, seguido significativamente por la **Temperatura Ambiente** y el **Índice UV**, factores que influyen directamente en la eficiencia del panel.
* **Varianza Espacial:** Se validó que los datos del aeropuerto requieren ajustes específicos del sitio para ser útiles en predicciones de alta fidelidad, debido a la nubosidad local y efectos térmicos.

## 📁 Estructura del Repositorio
* **`Jupyter/`**: Contiene el notebook principal `Caso_Estudio_Solar_Malambo.ipynb` con el flujo completo de análisis y ML.
* **`visuals/`**: Exportación en alta resolución de las 13 figuras académicas generadas (distribuciones, perfiles horarios, importancia de variables, evaluación visual de residuos, etc.).
* **`csv/`**: Dataset `ANALISIS_SOLAR_UNIFICADO.xlsx` con los datos originales y procesados.
* **`requirements.txt`**: Archivo de dependencias para reproducibilidad total del entorno.

---
**Autor:** Maria Jose Leal Brochero  
*Estudiante de la Maestría en Gestión Energética - Universidad del Atlántico*