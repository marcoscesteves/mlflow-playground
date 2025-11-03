# Controle de log
Há duas formas de exibir avisos durante a execução:
1. No console do Python (o terminal/saída do VS Code).
2. Como artifact no MLflow (fica salvo dentro do run).

# Por que registrar avisos no MLflow?
Enquanto as mensagens no console ajudam numa validação rápida durante a execução, as mensagens registradas no MLflow servem a vários propósitos. Alguns exemplos:

## Histórico durável
O que você escreve fica salvo ao lado de parâmetros, métricas e artefatos **daquele run**. O que apareceu no console se perde facilmente; o **artifact** no MLflow permanece e pode ser consultado depois.

## Reprodutibilidade
Dá para anotar detalhes de como a execução foi realizada, facilitando ao trabalho de quem queira reproduzir exatamente o que foi realizado.

## Contexto humano
Métricas dizem o que saiu; notas explicam por quê: hipótese, ajustes que você testou, bug observado, links da issue etc.

## Projeto contendo mensagens no console e MLFlow:


```diff
import mlflow
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

mlflow.set_experiment("02") 

+ with mlflow.start_run(run_name="teste02_duas_saidas"):
    # Params só para referência
    mlflow.log_params({"projeto": "teste-mlflow", "teste": "02"})

    # Mensagem que queremos ver nos dois lugares
    msg = "Executando Teste 02: mensagem aparece no console e no MLflow."

    # (1) Console/terminal (VS Code)
    print(msg)                 # saída simples
    logging.info(msg)          # saída com nível de log

    # (2) Artifact no MLflow (fica salvo no run)
+   mlflow.log_text(msg + "\nFonte: 02.py", "reports/notas.txt")

print("✔️ Finalizado. Abra a UI do MLflow para ver o artifact.")
```

Com relação ao projeto 01, devemos anotar algumas mudanças:

1. with mlflow.start_run(run_name="teste02_duas_saidas"):

Agora o run tem nome específico, em vez de um deixar a tarefa ao MLFlow, que cria um nome genérico;

2. mlflow.log_text(msg + "\nFonte: 02.py", "reports/notas.txt")
log_text registra uma nota como artifact dentro do MLflow (útil para contexto, reprodutibilidade e auditoria).
