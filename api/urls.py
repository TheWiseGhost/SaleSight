from django.urls import path
from .views import signup, login, verify, data, delete_row, edit_feature, models, get_feature_values_and_models, main, get_prediction, find_features, store_model, insert_row, add_feature, delete_feature, get_feature_values, recent_data, recent_models, user_info, export_csv

urlpatterns = [
    path('', main),
    path('sign_up/', signup, name='sign_up'),
    path('login/', login, name='login'),
    path('verify/<str:token>/', verify),
    path('prediction/', get_prediction),
    path('features/', find_features),
    path('store_model/', store_model),
    path('insert_row/', insert_row),
    path('delete_row/', delete_row),
    path('add_feature/', add_feature),
    path('delete_feature/', delete_feature),
    path('edit_feature/', edit_feature),
    path('get_feature_values/', get_feature_values),
    path('get_prediction_form/', get_feature_values_and_models),
    path('recent_data/', recent_data),
    path('data/', data),
    path('recent_models/', recent_models),
    path('models/', models),
    path('user_info/', user_info),
    path('export_csv/', export_csv),
]