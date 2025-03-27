from django.db import models

# Create your models here.

class HouseData(models.Model):
    area = models.FloatField()
    bedrooms = models.IntegerField()
    bathrooms = models.IntegerField()
    stories = models.IntegerField()
    mainroad = models.BooleanField()
    guestroom = models.BooleanField()
    basement = models.BooleanField()
    hotwaterheating = models.BooleanField()
    airconditioning = models.BooleanField()
    parking = models.IntegerField()
    prefarea = models.BooleanField()
    furnishingstatus = models.CharField(max_length=20)
    price = models.FloatField()

    def __str__(self):
        return f"House {self.id} - Price: {self.price}"

class PredictionResult(models.Model):
    TECHNIQUE_CHOICES = [
        ('polynomial', 'Polynomial Regression'),
        ('logistic', 'Logistic Regression'),
        ('knn', 'k-NN Classifier'),
    ]
    
    technique = models.CharField(max_length=20, choices=TECHNIQUE_CHOICES)
    input_data = models.JSONField()
    prediction = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.technique} Prediction - {self.created_at}"
