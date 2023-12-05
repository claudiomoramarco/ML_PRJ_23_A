import numpy as np
from typing import List
from tqdm import tqdm
import time
from layers import Layer, InputLayer
from activationfunctions import Activation
from metrics import Metric
from lossfunctions import Loss
from regularization import Regularizer
from stopping import Callback


""" Creiamo una barra di avanzamento che ci dica a che punto siamo"""
# Il formato per la barra di avanzamento
fmt = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}[{postfix}]"

@staticmethod
def update_bar(bar, stats):
    bar.set_postfix(stats)
    bar.update(1)