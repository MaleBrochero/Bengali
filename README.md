# Análisis de Potencia Fotovoltaica: Caso I.E. Nuestra Señora de la Candelaria

Este proyecto desarrolla un **Análisis Correlacional y Multivariado** para evaluar la precisión de la estimación de potencia en un sistema fotovoltaico on-grid ubicado en Malambo, Atlántico. 

El núcleo de la investigación compara los datos de generación real del inversor frente a las estimaciones teóricas basadas en datos climáticos del Aeropuerto Ernesto Cortissoz, identificando el impacto de la varianza microclimática local.

## 📊 Resumen del Análisis
Se implementó un pipeline de datos en Python para:
1. **Modelamiento Físico:** Cálculo de la generación teórica mediante el modelo de un diodo.
2. **Análisis de Residuos:** Identificación de brechas de potencia entre el modelo regional y la realidad local.
3. **Estadística Multivariada:** Uso de modelos OLS (Ordinary Least Squares) para cuantificar el impacto de la temperatura y la humedad en el error de estimación.

## 🛠️ Tecnologías Utilizadas
* **Lenguaje:** Python 3.13
* **Librerías principales:** * `pandas` & `numpy` para procesamiento de datos.
    * `pvlib` para simulación de sistemas solares.
    * `statsmodels` para la regresión multivariada.
    * `seaborn` & `matplotlib` para visualización avanzada.

## 📈 Hallazgos Clave
* **Impacto Térmico:** Se encontró una correlación positiva moderada ($r = 0.31$) entre la temperatura ambiente y el residuo de potencia.
* **Significancia:** El modelo OLS confirmó que la temperatura es un predictor crítico del error ($p < 0.001$), validando la necesidad de sensores locales frente a datos remotos.
* **Varianza Espacial:** El análisis sugiere que el 90% del error de estimación se debe a factores microclimáticos y nubosidad local no capturada por la estación meteorológica del aeropuerto.

## 📁 Estructura del Repositorio
* `Caso estudio .ipynb`: Notebook con el código completo, desde la limpieza hasta el modelo estadístico.
* `ANALISIS_SOLAR_UNIFICADO.xlsx`: Dataset enriquecido con columnas de potencia teórica y residuos calculados.

---
**Autor:** Maria Jose Leal Brochero  
*Estudiante de la Maestría en Gestión Energética - Universidad del Atlántico*