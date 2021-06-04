from os.path import join

import torch
import wandb

from torch.utils.data import DataLoader, TensorDataset
from src.utils.load_data import load_data
from src.model.get_model import get_multi_linear_residual_model
from src.utils.data_preprocess import get_data

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

