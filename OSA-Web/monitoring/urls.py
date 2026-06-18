from django.urls import path
from . import views

app_name = 'monitoring'

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('session/start/', views.start_session, name='start_session'),
    path('session/stop/', views.stop_session, name='stop_session'),
    path('sessions/', views.sessions_list, name='sessions'),
    path('sessions/<int:pk>/', views.session_detail, name='session_detail'),
    path('analytics/', views.analytics, name='analytics'),
    path('inventory/', views.inventory, name='inventory'),
    path('settings/', views.settings_view, name='settings'),
    path('settings/camera/<int:pk>/delete/', views.delete_camera, name='delete_camera'),
    path('settings/alert/<int:pk>/delete/', views.delete_alert, name='delete_alert'),
    path('export/<str:dataset>/', views.export_table, name='export_table'),
    path('sessions/<int:pk>/export/<str:dataset>/', views.export_session_table, name='export_session_table'),
    
    # AI Co-pilot routes
    path('copilot/', views.copilot, name='copilot'),
    path('api/copilot/chat/', views.copilot_chat_api, name='copilot_chat_api'),
    path('api/copilot/clear/', views.clear_chat_history, name='clear_chat_history'),
]
