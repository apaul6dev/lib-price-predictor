# Crear entorno virtual si no existe
python3 -m venv .venv

# Activarlo en Mac/Linux
source .venv/bin/activate

# Activarlo en Windows (cmd)
.venv\Scripts\activate

# Crear la libreria
pip install -e . 
# probar la libreria
python run_experiment.py

# Instalar en modo editable (desarrollo)
pip install -e .

# Crear una distribución en formato wheel
python -m build

# Verificar que se incluyeron los archivos
tar -tzf dist/vehicle_price_predictor-0.1.0.tar.gz

# Descripción de datos de entrada

| Columna                    | Tipo            | Descripción                                                               |
| -------------------------- | --------------- | ------------------------------------------------------------------------- |
| `Unnamed: 0`               | `int`           | Índice original del dataset. Generalmente no es necesario para el modelo. |
| `brand`                    | `str`           | Marca del vehículo en minúsculas (ej. `ford`, `audi`, `hyundai`).         |
| `model`                    | `str`           | Nombre y línea del modelo (ej. `Ford Kuga`, `Audi Q4 e-tron`).            |
| `color`                    | `str`           | Color exterior del vehículo (ej. `black`, `grey`, `red`).                 |
| `registration_date`        | `str` (MM/YYYY) | Fecha de primera matriculación del vehículo (ej. `05/2023`).              |
| `year`                     | `int`           | Año de fabricación del vehículo.                                          |
| `price_in_euro`            | `float`         | Precio de venta del vehículo en euros. **Variable objetivo (target).**    |
| `power_kw`                 | `int`           | Potencia del motor en kilovatios (kW).                                    |
| `power_ps`                 | `int`           | Potencia del motor en caballos de fuerza (PS).                            |
| `transmission_type`        | `str`           | Tipo de transmisión: `Manual` o `Automatic`.                              |
| `fuel_type`                | `str`           | Tipo de combustible: `Petrol`, `Diesel`, `Hybrid`, `Electric`, etc.       |
| `fuel_consumption_l_100km` | `str`           | Consumo de combustible en litros cada 100 km (ej. `"5,4 l/100 km"`).      |
| `fuel_consumption_g_km`    | `str`           | Emisiones de CO₂ en gramos por kilómetro (ej. `"124 g/km"`).              |
| `mileage_in_km`            | `float`         | Kilometraje total del vehículo (ej. `57000.0`).                           |
| `offer_description`        | `str`           | Texto libre que describe características adicionales de la oferta.        |
