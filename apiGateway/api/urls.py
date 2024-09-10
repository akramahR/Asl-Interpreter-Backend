from django.urls import path
from .views import *

urlpatterns = [
    path('predict/', Prediction.as_view(), name = 'predict'),
    path('upload/', upload_page, name='upload_page'),
]