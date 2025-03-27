from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('polynomial/', views.polynomial, name='polynomial'),
    path('logistic/', views.logistic, name='logistic'),
    path('knn/', views.knn, name='knn'),
] 