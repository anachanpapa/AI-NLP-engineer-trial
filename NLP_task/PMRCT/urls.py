from django.conf.urls import include, url
from . import views

urlpatterns = [
    url(r'^PMRCT/$', views.bootstrapping, name='bootstrapping'),
    url(r'^PMRCT/input_seed_data/$', views.input_seed_data, name='input_seed_data'),
    url(r'^PMRCT/search_new_snippets/$', views.search_new_snippets, name='search_new_snippets'),
    url(r'^PMRCT/snippets_classification/$', views.snippets_classification, name='snippets_classification'),
    url(r'^PMRCT/register_new_words/$', views.register_new_words, name='register_new_words'),
    url(r'^PMRCT/register_new_phrases/$', views.register_new_phrases, name='register_new_phrases'),
    url(r'^PMRCT/active_learning/$', views.active_learning, name='active_learning'),
]