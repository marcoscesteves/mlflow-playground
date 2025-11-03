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

    # Salvando o "artifact"
    import pandas as pd
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    df.to_csv("dados.csv")
    mlflow.log_artifact("dados.csv")
    
   # Salvando um gráfico
    import matplotlib.pyplot as plt
    plt.plot([1, 2, 3], [4, 5, 6])
    plt.savefig("grafico.png")
    mlflow.log_artifact("grafico.png")
