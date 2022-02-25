from django.urls import path

from . import views

app_name = 'benchmark'
urlpatterns = [
    path('', views.IndexView.as_view(), name='index'),

    path('score/', views.score, name='score'),
    path('run/', views.run, name='run'),
    path('evaluate/', views.evaluate, name='evaluate'),
    path('compare/', views.compare, name='compare'),
    path('signin/', views.signin, name='signin'),
    path('signout/', views.signout, name='signout'),
    path('list/', views.ListView.as_view(), name='list'),

    # ex: /benchmark/5/
    path('<int:pk>/', views.DetailView.as_view(), name='detail'),

    # ex: /benchmark/5/results/
    path('<int:pk>/results/', views.ResultsView.as_view(), name='results'),

    # ex: /benchmark/5/vote/
    path('<int:question_id>/vote/', views.vote, name='vote'),
]
