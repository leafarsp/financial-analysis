import subprocess
import sys
import os

# Obtém o caminho do arquivo requirements.txt
requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")

# Verifica se o arquivo requirements.txt existe
if not os.path.exists(requirements_path):
    print(f"Erro: O arquivo {requirements_path} não foi encontrado!")
    sys.exit(1)

# Executa o pip install -r requirements.txt
try:
    print("Instalando dependências...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])
    print("Instalação concluída com sucesso!")
except subprocess.CalledProcessError as e:
    print(f"Erro ao instalar as dependências: {e}")
    sys.exit(1)
