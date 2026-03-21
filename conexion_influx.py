import influxdb_client
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

# === CONFIGURACIÓN ===
TOKEN = "0Ua_yjE85wlCYgllphsTIWSjd5CyRSEn3UuPC8abGYvi4HE8N7S1nMmcwS1QP7jq7ithq_hGVR8U7ELkVni-IA==" 
ORG = "estacion_bengali"
BUCKET = "estacion_bengali"
URL = "http://192.168.1.XX:8086" # Reemplaza con la IP de tu Raspberry

client = InfluxDBClient(url=URL, token=TOKEN, org=ORG)
write_api = client.write_api(write_options=SYNCHRONOUS)

def enviar_datos_vantage_pro2(datos_davis):
    """
    Envía el paquete completo de una Davis Vantage Pro2 a InfluxDB.
    """
    try:
        punto = Point("archivo_vantage_pro2") \
            .tag("estacion", "bengali_vantage_pro2") \
            .tag("modelo", "Vantage_Pro2") \
            .field("temp_out", float(datos_davis['temp'])) \
            .field("hum_out", float(datos_davis['hum'])) \
            .field("bar_press", float(datos_davis['bar'])) \
            .field("wind_speed", float(datos_davis['wind_s'])) \
            .field("wind_dir", float(datos_davis['wind_d'])) \
            .field("rain_rate", float(datos_davis['rain'])) \
            .field("solar_rad", float(datos_davis['solar'])) \
            .field("uv_index", float(datos_davis['uv']))

        write_api.write(bucket=BUCKET, org=ORG, record=punto)
        print(f"✅ [DAVIS] Paquete completo enviado a la base de datos.")
        
    except Exception as e:
        print(f"❌ Error al enviar datos Davis: {e}")

if __name__ == "__main__":
    # Estos son los datos típicos que entrega una Vantage Pro2:
    datos_simulados = {
        'temp': 26.5, 'hum': 75.0, 'bar': 1012.5, 
        'wind_s': 12.4, 'wind_d': 180, 'rain': 0.0,
        'solar': 850.0, 'uv': 5.5
    }
    
    enviar_datos_vantage_pro2(datos_simulados)
    client.close()