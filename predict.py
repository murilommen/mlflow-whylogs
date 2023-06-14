import mlflow
import pandas as pd

logged_model = 'runs:/215cedb88ff347f8923724d8e124df09/model'
loaded_model = mlflow.pyfunc.load_model(logged_model)

data = pd.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4]})
print(loaded_model.predict(pd.DataFrame(data)))