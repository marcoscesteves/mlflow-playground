# MLflow - Guia Básico

Nesta primeiro tutorial vamos inicial bem do básico sobre o MLFLow.

## 1. Como Inicializar o MLflow

Para começar, você precisa:

1. Instalar o MLflow:
```bash
pip install mlflow
```

2. Iniciar o servidor MLflow UI (interface web):
```bash
mlflow ui
```
O servidor será iniciado em `http://localhost:5000`

![alt text](/img/tela1.jpeg)

**IMPORTANTE**: Ao ser executado, o MLflow cria uma pasta de controle (por padrão chamada `mlruns`) no diretório atual, onde são armazenados logs, parâmetros, métricas e artefatos. Se você iniciar o MLflow  (`mlflow ui`) em outro diretório, a ferramenta **não exibirá** os dados que foram armazenados em outro local.

Repare neste exemplo abaixo:

![alt text](/img/tela2.jpeg)

Repare que a pasta atual possui o diretório `mlruns`. É exatamente dentro desta pasta que ficam armazenados os runs que o MLFLow coleta. Repare na representação em árvore da pasta:

![alt text](/img/imgMlRuns.jpeg)

Mais pra frente veremos que é possível executar o MLflow a partir de qualquer diretório. Mas isso foge do escopo desta primeira atividade.

## 2. Conceitos e Exemplos Básicos

### Importar o MLFLow em seu projeto
Abrir uma IDE ou editor simples de texto e criar a estrutura do projeto que vamos mapear usando o mlflow. A primeira linha do código deve ser o "import" do MLFlow.

```python
import mlflow
```

### Definir o nome do experimento 
Define o nome do experimento onde suas execuções (runs) vão ficar agrupadas.
```python
mlflow.set_experiment("meu-experimento")
```

### Avisar onde o MLFlow comecará a observar

```python
with mlflow.start_run():
    # seu código aqui
```

### Registrar parâmetros do seu experimento:

```diff
import mlflow
mlflow.set_experiment("01")
with mlflow.start_run():
    # Registrando parâmetros individuais
+   mlflow.log_param("model_name", "baseline")
+   mlflow.log_param("learning_rate", 0.01)
```
No exemplo acima estamos informando ao MLflow a existência de dois parâmetros: model_name e learning_rate. Se preferirmos poderíamos passar esses parâmetros de forma conjugada.

```diff
import mlflow
mlflow.set_experiment("01")
with mlflow.start_run():
    # Registrando múltiplos parâmetros
+    params = {
+        "model_name": "baseline",
+        "learning_rate": 0.01,
+    }
+
+    mlflow.log_params(params)
```
### Métricas do seu experimento/modelo
Exemplos: accuracy, f1, rmse, auc, loss, etc.
```diff
import mlflow
mlflow.set_experiment("01")
with mlflow.start_run():
    # Registrando múltiplos parâmetros
    params = {
        "model_name": "baseline",
        "learning_rate": 0.01,
    }

    mlflow.log_params(params)

    # Registrar alguma métrica só para testar
+   mlflow.log_metric("passos", 1)
+   mlflow.log_metric("acuracia_fake", 0.0)
```


### Artefatos (Artifacts)
São arquivos anexados à sua execução. Podem ser datasets, modelos, imagens, gráficos, arquivos csv etc.

```diff
import mlflow
mlflow.set_experiment("01")
with mlflow.start_run():
    # Registrando múltiplos parâmetros
    params = {
        "model_name": "baseline",
        "learning_rate": 0.01,
    }

    mlflow.log_params(params)

    # Registrar alguma métrica só para testar
    mlflow.log_metric("passos", 1)
    mlflow.log_metric("acuracia_fake", 0.0)

+   # Salvando um DataFrame
+   import pandas as pd
+   df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
+   df.to_csv("dados.csv")
+   mlflow.log_artifact("dados.csv")
    
+   # Salvando um gráfico
+   import matplotlib.pyplot as plt
+   plt.plot([1, 2, 3], [4, 5, 6])
+   plt.savefig("grafico.png")
+   mlflow.log_artifact("grafico.png")
```

