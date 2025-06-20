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