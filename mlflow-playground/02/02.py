import mlflow
from pathlib import Path
import logging

# Configura logging para o console (saída do VS Code) 
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

mlflow.set_experiment("02")

with mlflow.start_run(run_name="teste02_duas_saidas"):
    # Params só para referência
    mlflow.log_params({"projeto": "teste-mlflow", "teste": "02"})

    # Mensagem que queremos ver nos dois lugares
    msg = "Executando Teste 02: mensagem aparece no console e no MLflow."

    # (1) Console/terminal (VS Code)
    print(msg)                 # saída simples
    logging.info(msg)          # saída com nível de log

    # (2) Artifact no MLflow (fica salvo no run)
    mlflow.log_text(msg + "\nFonte: 02.py", "reports/notas.txt")

print("✔️ Finalizado. Abra a UI do MLflow para ver o artifact.")