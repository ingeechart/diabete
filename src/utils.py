import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


_MIN = 0
_MAX = 300

class Normalization(object):
    def __init__(self, min = _MIN, max = _MAX):
        super().__init__()
        self.min = min
        self.max = max

    def normalize(self,data):
        # x- min/ max-min
        norm_data = (data-self.min) / (self.max - self.min)
        return norm_data

    def de_normalize(self, data):
        de_norm_data = data*(self.max-self.min)+self.min
        return de_norm_data
    
