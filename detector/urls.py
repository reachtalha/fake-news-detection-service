from django.urls import path
from .views import FakeNewsView

urlpatterns = [
    path('detect_news', FakeNewsView.as_view(), name='detect_news'),
]
