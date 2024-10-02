from django.apps import AppConfig


#class ApiConfig(AppConfig):
    #default_auto_field = "django.db.models.BigAutoField"
    #name = "api"

import os
from django.apps import AppConfig
from django.conf import settings
import numpy as np
import torch
import pickle

class ApiConfig(AppConfig):
    name = 'api'

    MODEL_FILE = os.path.join(settings.MODELS, "simple_nn_model.pth")
    model = torch.load(MODEL_FILE)

    LABEL_FILE = os.path.join(settings.MODELS, "label_encoder.pkl")
    with open(LABEL_FILE, 'rb') as f:
        label_encoder = pickle.load(f)

