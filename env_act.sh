# Windows PowerShell activation and installation script for .rainflow virtual environment
python -m venv .rainflow
. .rainflow\Scripts\Activate
python -m pip install --upgrade pip
pip install -r requirements.txt
