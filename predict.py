import mlflow
import pandas as pd

logged_model = 'runs:/8772de8e19e84d2fb43ecdbe2100f602/model'
loaded_model = mlflow.pyfunc.load_model(logged_model)

data = pd.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4]})
print(loaded_model.predict(pd.DataFrame(data)))