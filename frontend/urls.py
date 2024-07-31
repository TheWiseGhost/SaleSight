from django.urls import path
from .views import index

urlpatterns = [
    path('', index),
    path('dashboard/', index),
    path('dataportal/', index),
    path('aiportal/', index),
    path('data/', index),
    path('models/', index),
    path('landing/', index),
    path('auth/', index),
    path('profile/', index),
    path('chooseplan/', index),
    path('verify/<str:token>/', index),
]
