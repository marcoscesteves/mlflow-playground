# Trabalhando com múltiplas execuções no MLFlow
É possível controlar variadas execuções dentro do MLFlow. 



```diff
import mlflow
import math
import random

# 1) Definir o experimento 
mlflow.set_experiment("03")

# 2) Vamos simular algumas combinações de parâmetros
configs = [
    {"model_name": "baseline", "learning_rate": 0.01},
    {"model_name": "baseline", "learning_rate": 0.02},
    {"model_name": "baseline", "learning_rate": 0.05},
]

for i, cfg in enumerate(configs, start=1):
    # 3) Cada iteração é um run separado
    with mlflow.start_run(run_name=f"run_{i:02d}"):
        # params
        mlflow.log_params(cfg)
        mlflow.set_tag("atividade", "03")
        mlflow.set_tag("tópico", "loop")

        # 4) Simular uma métrica que depende do parâmetro
        # (só pra ter número diferente)
        lr = cfg["learning_rate"]
        fake_loss = round(1 / (1 + lr * 100) + random.uniform(0, 0.05), 4)
        fake_acc = round(0.6 + lr * 5 + random.uniform(0, 0.02), 4)

        mlflow.log_metric("fake_loss", fake_loss)
        mlflow.log_metric("fake_acc", fake_acc)

        # 5) Mensagenzinha pra console
        print(f"[Atividade 03] Finalizado run {i} com lr={lr}")

print("✅ Atividade 03 concluída. Abra o MLflow e compare os runs do experimento '03'.")
```

Alguns pontos que devem ser observados neste código:

1. Configs

Definimos uma lista (chamada configs) que possui três elementos. O primeiro elemento seria: 

{"model_name": "baseline", "learning_rate": 0.01}

O segundo elemento seria: 

{"model_name": "baseline", "learning_rate": 0.02}

E o terceiro elemento seria:

{"model_name": "baseline", "learning_rate": 0.05}.

Por fim, vale mencionar que cada elemento desse (envoltos por chaves e composto por pares de chave-valor) são - na verdade - dicionários.

2. for i, cfg in enumerate(configs, start=1):

Aqui temos um loop usando enumerate. Entender o enumerate é muito útil em Python, motivo pelo qual vamos explicar neste ponto.
A função enumerate em Python transforma a lista configs em pares (tuplas) onde cada par contém:

- O índice (posição) do item na sequência.
- O próprio item da sequência.

Então, por exemplo, no loop acima durante a primeira iteração teremos: 
- i: 1
- elemento: ({"model_name": "baseline", "learning_rate": 0.01})

3. O objetivo desta atividade é notar as várias execuções dentro do MLFlow e a organização realizada na ferramenta.