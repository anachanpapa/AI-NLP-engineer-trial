from django.contrib import admin
from .models import TrainingData, Behavior, Change, Middle, Regex

admin.site.register(TrainingData)
admin.site.register(Behavior)
admin.site.register(Change)
admin.site.register(Middle)
admin.site.register(Regex)
