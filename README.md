# Useful commands
- python3 train.py # outputs run_uuid

run_uuid example: 368d6908c08447589892e2634eb5fc25

Build docker-image:

mlflow models build-docker --name "test-whylogs-mlflow" -m "runs:/368d6908c08447589892e2634eb5fc25/model"