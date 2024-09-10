from django.apps import AppConfig


#class ApiConfig(AppConfig):
    #default_auto_field = "django.db.models.BigAutoField"
    #name = "api"

import os
from django.apps import AppConfig
from django.conf import settings
import numpy as np
from tensorflow.keras.models import load_model


class ApiConfig(AppConfig):
    name = 'api'

    MODEL_FILE = os.path.join(settings.MODELS, "asl_gesture_recognition_model.keras")
    model = load_model(MODEL_FILE)

    LABEL_FILE = os.path.join(settings.MODELS, "label_classes.npy")
    label_classes = np.load(LABEL_FILE)
