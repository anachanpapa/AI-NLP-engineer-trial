# -*- coding: utf-8 -*-
# Generated by Django 1.11.5 on 2017-09-21 14:19
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('PMRCT', '0005_delete_regex'),
    ]

    operations = [
        migrations.CreateModel(
            name='Regex',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('expression', models.TextField()),
            ],
            options={
                'ordering': ['expression'],
            },
        ),
        migrations.DeleteModel(
            name='Snippet',
        ),
    ]
