""" entrypoint to BicycleGAN model. Deprecated.

MLflow runner is initially setup as in
https://www.mlflow.org/docs/latest/tutorials-and-examples/tutorial.html#training-the-model
"""

import argparse
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml

from experiments import get_eie_experiment
from src.models import eie_models


# def eval_metrics(actual, pred):
#     rmse = np.sqrt(mean_squared_error(actual, pred))
#     mae = mean_absolute_error(actual, pred)
#     r2 = r2_score(actual, pred)
#     return rmse, mae, r2


def setup_args():
    """ Combine params from command line and yaml config.

    Returns
    -------
    config : dict

    """
    parser = argparse.ArgumentParser(description='Generic runner for our models')
    parser.add_argument('--config', '-c',
                        dest="config_fin",
                        metavar='FILE',
                        help='path to the config file',
                        default='configs/gan.yaml')

    # Experiment params
    parser.add_argument('--dataset', help='dataset name',
        dest='exp_params.dataset', default=None, type=str)
    parser.add_argument('--batch_size', help='batch size',
        dest='exp_params.batch_size', default=None, type=int)
    parser.add_argument('--lr', help='learning rate',
        dest='exp_params.LR', default=None)

    # Trainer params
    parser.add_argument('--gpus', help='batch size',
        dest='trainer_params.gpus', default=None)
    parser.add_argument('--max_epochs', help='batch size',
        dest='trainer_params.max_epochs', default=None, type=int)

    args = parser.parse_args()
    with open(args.config_fin, 'r') as fin:
        try:
            config = yaml.safe_load(fin)
        except yaml.YAMLError as exc:
            print(exc)

    for k, v in vars(args).items():
        if "." in k and v:
            k1, k2 = k.split('.')
            config[k1][k2] = v

    return config


def get_model(model_params):
    """Lookup the model class to use; expected to be importable from src.models.eie_models.

    Params
    ------
    model_params : dict
        Model params entry of the runner config, which must have a known "name" entry.

    Returns
    -------
    model : pytorch.model

    Raises
    ------
    RuntimeError for bad model config

    """
    try:
        model = eie_models[model_params['name']](**model_params)
    except KeyError as e:
        raise RuntimeError("Failed to load model from `config`: {}".format(e))
    return model


if __name__ == "__main__":

    config = setup_args()

    # For reproducibility
    torch.manual_seed(config['logging_params']['manual_seed'])
    np.random.seed(config['logging_params']['manual_seed'])
    cudnn.deterministic = True
    cudnn.benchmark = False

    model = get_model(config['model_params'])
    experiment = get_eie_experiment(model, config['exp_params'])

    with mlflow.start_run():
        model.fit(train_x, train_y)

        predicted_qualities = model.predict(test_x)
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        # Log the params that we care about
        mlflow.log_param("LR", config['exp_params']['LR'])
        mlflow.log_param("weight_decay", config['exp_params']['weight_decay'])
        mlflow.log_param("scheduler_gamma", config['exp_params']['scheduler_gamma'])
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

    print(f"======= Starting {config['model_params']['name']} =======")
    experiment.run()

    # Cleanup temp expeiment folder
    shutil.rmtree(experiment.out_dir)
