# Bengali
# 🌿 Estación Bengalí - Monitoreo Ambiental IoT

Este proyecto consiste en una estación de monitoreo de datos técnicos y ambientales, diseñada para recolectar, almacenar y visualizar datos durante un periodo de **7 meses**. El sistema utiliza una **Raspberry Pi** como servidor local corriendo contenedores de **Docker**.

## 🚀 Arquitectura del Sistema
* **Hardware:** Raspberry Pi (Servidor).
* **Base de Datos:** [InfluxDB v2](https://www.influxdata.com/) ejecutándose en un contenedor Docker.
* **Lenguaje:** Python 3.
* **Visualización:** Dashboards integrados de InfluxDB.

## 📊 Configuración de la Base de Datos
La base de datos está optimizada para el almacenamiento a largo plazo con las siguientes especificaciones:
* **Organización:** `estacion_bengali`
* **Bucket:** `estacion_bengali`
* **Política de Retención:** 1 año (365 días).
* **Seguridad:** Acceso mediante API Tokens de nivel "All Access".

## 🛠️ Instalación y Requisitos

Para ejecutar el script de recolección de datos, es necesario instalar la librería cliente de InfluxDB:

```bash
pip install influxdb-client
