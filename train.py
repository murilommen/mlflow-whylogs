import json

import mlflow
import pandas as pd
import cloudpickle
import pickle
import requests
import whylogs as why
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class MyModelWrapper(mlflow.pyfunc.PythonModel):

    def __init__(self):
        self.preprocessor = MinMaxScaler()
        self.model = RandomForestClassifier()
        self.logger = None

    def _start_logger(self):
        self.logger = why.logger(mode="rolling", interval=5, when="M",
                                base_name="message_profile")

        self.logger.append_writer("local", base_dir="example_output")
    
    def __del__(self):
        # On exit the rest of the logging will be saved
        if self.logger:
            self.logger.close()
    
    def load_context(self, context):
        with open(context.artifacts["preprocessor"], "rb") as f:
            self.processor = pickle.load(f)
        with open(context.artifacts["estimator"], "rb") as f:
            self.estimator = pickle.load(f)

    def fit(self, data: pd.DataFrame) -> None:
        # Making the splits
        X = data.drop('target', axis=1)
        y = data['target']

        # Training the transformer and scaling the data
        X_scaled = self.preprocessor.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25)

        # Training your model Pipeline
        self.model.fit(X_train, y_train)

        # Evaluating and logging metric with MLFlow
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)

        # Dumping the fitted objects
        with open("fitted_processor.pkl", "wb") as f:
            cloudpickle.dump(self.preprocessor, f)

        with open("fitted_model.pkl", "wb") as f:
            cloudpickle.dump(self.model, f)

    def predict(self, context, data):
        if not self.logger:
            self._start_logger()
        
        transformed_data = self.preprocessor.transform(data)
        predictions = self.model.predict(transformed_data)
        
        
        df = pd.DataFrame(data)
        df["output_model"] = predictions
        # response = requests.post(
        #     url="http://localhost:8080/logs",
        #     data=json.dumps({
        #             "datasetId": "model-5",
        #             "timestamp": 0,
        #             "multiple": df.to_dict(orient="split")
        #         })
        # )
        # print(response)
        
        self.logger.log(df)
        
        return predictions


def read_data() -> pd.DataFrame:
    data = load_iris()
    df = pd.DataFrame(data.data)
    df["target"] = data.target
    return df


with mlflow.start_run() as run:

    model = MyModelWrapper()

    data = read_data()

    model.fit(data)

    artifacts = {
        "preprocessor" : "fitted_processor.pkl",
        "estimator" : "fitted_model.pkl"
    }

    mlflow.pyfunc.log_model(
        artifact_path='model',
        artifacts=artifacts,
        python_model = model,
        conda_env = None,
        pip_requirements=["sklearn==1.2.2", "mlflow=2.3.2", "dill", "pandas", "whylogs[whylabs]"]
    )

    print(run.info.run_id)
    
    
    mlflow.end_run()

