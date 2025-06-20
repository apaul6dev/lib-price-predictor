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