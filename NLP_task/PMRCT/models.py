from django.db import models
from django.utils import timezone

class TrainingData(models.Model):
    POS = 1
    NEG = 0
    UNK = 2
    CLASS_CHOICES = (
        (POS, 'pos'),
        (NEG, 'neg'),
        (UNK, 'unk'),
    )
    expression = models.CharField(max_length=300)
    isBehavior = models.IntegerField(choices=CLASS_CHOICES)

    def __str__(self):
        return '{0}: {1}'.format(self.isBehavior, self.expression)

    class Meta:
        ordering = ["isBehavior", "expression"]


class Behavior(models.Model):
    USED = 1
    UNUSED = 0
    USED_CHOICES = (
        (USED, 'used'),
        (UNUSED, 'unused'),
    )    
    expression = models.CharField(max_length=100)
    isUsed = models.IntegerField(choices=USED_CHOICES)

    def __str__(self):
        return '{0}: {1}'.format(self.isUsed, self.expression)

    class Meta:
        ordering = ["isUsed", "expression"]


class Change(models.Model):
    USED = 1
    UNUSED = 0
    USED_CHOICES = (
        (USED, 'used'),
        (UNUSED, 'unused'),
    )    
    expression = models.CharField(max_length=100)
    isUsed = models.IntegerField(choices=USED_CHOICES)

    def __str__(self):
        return '{0}: {1}'.format(self.isUsed, self.expression)

    class Meta:
        ordering = ["isUsed", "expression"]       


class Middle(models.Model):
    USED = 1
    UNUSED = 0
    UNUSABLE = 2
    USED_CHOICES = (
        (USED, 'used'),
        (UNUSED, 'unused'),
        (UNUSABLE, 'unusable'),
    )    
    expression = models.CharField(max_length=100)
    isUsed = models.IntegerField(choices=USED_CHOICES)

    def __str__(self):
        return '{0}: {1}'.format(self.isUsed, self.expression)

    class Meta:
        ordering = ["isUsed", "expression"]  



class Regex(models.Model):
    expression = models.TextField()
    position_part = models.CharField(max_length=10, default='dummy')
    def __str__(self):
        return '{0}: {1}'.format(self.position_part, self.expression)

    class Meta:
        ordering = ["position_part", "expression"]  

