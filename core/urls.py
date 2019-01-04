from django.urls import path
from .views import upload_data, plots

urlpatterns = [
    path('', upload_data, name='upload_data'),
    path('plots/', plots, name='plots'),
]
