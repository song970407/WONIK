import torch
import wandb
from box import Box


def get_state_dict_and_config(wandb_run_path,
                              config_file_name: str = 'exp_config.yaml',
                              model_file_name: str = 'model.pt'):
    ret = dict()
    if config_file_name is not None:
        model_config_path = wandb.restore(config_file_name,
                                          wandb_run_path,
                                          replace=True)
        config = Box.from_yaml(filename=model_config_path.name)
        ret['config'] = config

    if model_file_name is not None:
        model_path = wandb.restore(model_file_name,
                                   wandb_run_path,
                                   replace=True)
        model_state_dict = torch.load(model_path.name, map_location='cpu')
        ret['state_dict'] = model_state_dict

    return ret
