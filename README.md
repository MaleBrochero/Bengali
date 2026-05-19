# Análisis Multivariado de Planta Fotovoltaica
## Institución Educativa – Malambo, Atlántico (Colombia)

| Campo | Detalle |
|---|---|
| **Programa** | Maestría en Gestión Energética – Universidad del Atlántico |
| **Sistema fotovoltaico** | 4 × Trina Solar TSM-540 DEG18MC.20(II) |
| **Inversor** | Microinversor Hoymiles (monitoreo en la nube) |
| **Proxy meteorológico** | Aeropuerto Internacional Ernesto Cortissoz (IATA: BAQ) |
| **Período de análisis** | 17 de abril – 08 de mayo de 2026 (22 días) |
| **Total de registros** | 1 136 (frecuencia mixta: 15 min predominante) |

---

## 🔬 Pregunta de Investigación

> *¿En qué medida las variables meteorológicas registradas en el Aeropuerto Ernesto Cortissoz (temperatura y humedad relativa) explican la diferencia entre la potencia teórica estimada mediante el modelo físico del diodo y la potencia real medida por el inversor Hoymiles en la Institución Educativa de Malambo?*

---

## 📊 Resumen del Pipeline Metodológico

El flujo de trabajo implementado en el ecosistema de Python se divide en **12 secciones analíticas**:

1. **Configuración e Importaciones** – Instalación condicional de dependencias y configuración del entorno de visualización.
2. **Carga y Exploración de Datos (EDA)** – Estadísticas descriptivas y control de calidad (nulos, anomalías físicas).
3. **Pruebas de Normalidad** – Shapiro-Wilk, Anderson-Darling y Q-Q plots para las variables clave.
4. **Modelo Físico del Diodo Único (pvlib)** – Simulación de la potencia teórica mediante el modelo De Soto (2006).
5. **Validación: Real vs. Teórico** – Comparativa multi-día y análisis de sincronización de la brecha de potencia.
6. **Análisis Correlacional: Pearson y Spearman** – Heatmap y scatter de correlación entre variables.
7. **Regresión OLS Multivariada** – Modelo lineal de referencia con diagnóstico de heterocedasticidad (Breusch-Pagan).
8. **Métricas de Error del Modelo** – Comparativa de MAE, RMSE y R² para todos los modelos.
9. **Diagnóstico de Residuos** – Análisis de la distribución y comportamiento de los residuos del modelo OLS.
10. **Modelado Avanzado: Machine Learning No Lineal** – Random Forest y CatBoost Regressor con importancia de variables.
11. **Evaluación Visual: Residuos y Series Temporales** – Real vs. predicho (ML) y serie temporal continua (22 días).
12. **Conclusiones y Recomendaciones** – Síntesis de hallazgos y limitaciones metodológicas.

---

## 📈 Métricas de Rendimiento y Hallazgos Clave

| Modelo | MAE (W) | R² |
|---|---|---|
| Modelo Físico (Diodo) | 173.06 | — |
| Regresión Lineal (OLS) | 242.06 | 0.08 |
| Random Forest | 125.79 | 0.556 |
| **CatBoost Regressor** | **123.57** | **0.580** |

- **Salto Predictivo Radical:** La regresión lineal (OLS) es incapaz de modelar la complejidad del sistema. El modelo no lineal **CatBoost** incrementó drásticamente la capacidad explicativa.
- **Importancia de Variables:** La **Irradiancia Proxy** es el factor preponderante, seguida por la **Temperatura Ambiente** y el **Índice UV**.

---

## 🛠️ Tecnologías y Librerías Utilizadas

- **Entorno:** Python 3.13 + JupyterLab
- **Librerías clave:**
  - `pvlib` – Simulación física del panel solar (modelo De Soto del diodo único).
  - `statsmodels` – Regresión OLS y prueba de heterocedasticidad (Breusch-Pagan).
  - `scikit-learn` – Random Forest Regressor, métricas de validación y split de datos.
  - `catboost` – Gradient Boosting no lineal optimizado.
  - `scipy` – Pruebas de normalidad (Shapiro-Wilk, Anderson-Darling).
  - `pandas` & `numpy` – Procesamiento y limpieza de datos.
  - `seaborn` & `matplotlib` – Generación y exportación de visuales (300 DPI).
  - `openpyxl` – Lectura del archivo Excel de datos unificados.

---

## 📊 Catálogo de Figuras Académicas (`visuals/`)

El repositorio contiene **12 figuras académicas** exportadas en alta resolución (300 DPI):

| Figura | Descripción |
|---|---|
| `fig1_distribucion_variables.png` | Histogramas y KDE de distribución de las variables principales del dataset. |
| `fig2_perfil_horario.png` | Perfil horario promedio (boxplot por hora) de potencia real e irradiancia. |
| `fig3_heatmap_correlacion.png` | Mapa de calor de correlaciones de Pearson y Spearman entre todas las variables. |
| `fig4_analisis_irradiancia.png` | Boxplot de irradiancia estimada por franja horaria diurna. |
| `fig5_qq_normalidad.png` | Q-Q plots de normalidad para potencia real, temperatura y residuos. |
| `fig6_multidia_real_teorica.png` | Comparativa multi-día (6 fechas) de potencia real vs. teórica con brecha sombreada. |
| `fig7_multidia_sincronizacion.png` | Análisis multi-día de la sincronización temporal entre real y teórico. |
| `fig8_scatter_correlacion.png` | Scatter de correlación: potencia real vs. potencia teórica con línea de ajuste. |
| `fig9_diagnostico_residuos.png` | Diagnóstico completo de residuos del modelo OLS. |
| `fig10_importancia_variables.png` | Importancia relativa de variables en los modelos de Machine Learning. |
| `fig11_real_vs_predicho_ml.png` | Scatter comparativo: potencia real vs. predicciones de Random Forest y CatBoost. |
| `fig12_comparativa_mae.png` | Gráfico de barras comparativo del MAE (W) para todos los modelos analizados. |

> **Nota:** `fig15_serie_temporal_completa.png` (serie temporal de 22 días) se genera al ejecutar el notebook completo.

---

## 📁 Estructura del Repositorio

```
Bengali/
├── Jupyter/
│   └── Caso_Estudio_Solar_Malambo.ipynb   ← Notebook principal (completamente ejecutado)
├── visuals/                               ← 12 figuras académicas en PNG (300 DPI)
├── csv/
│   └── ANALISIS_SOLAR_UNIFICADO.xlsx      ← Dataset unificado (1 136 registros)
├── docs/                                  ← Presentación y documentos adicionales
├── requirements.txt                       ← Dependencias del entorno Python
└── README.md
```

---

## ▶️ Reproducibilidad

```bash
# 1. Clonar el repositorio
git clone https://github.com/MaleBrochero/Bengali.git
cd Bengali

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar el notebook
jupyter lab Jupyter/Caso_Estudio_Solar_Malambo.ipynb
```

> El notebook instala automáticamente las dependencias faltantes al ejecutarse.

---

**Autor:** Maria Jose Leal Brochero  
*Estudiante de la Maestría en Gestión Energética – Universidad del Atlántico*