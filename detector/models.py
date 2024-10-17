from django.db import models


class News(models.Model):
    class Meta:
        db_table = 'news'

    id = models.AutoField(primary_key=True)
    input_text = models.CharField(max_length=2500)
    text_length = models.CharField(max_length=20)
    genuine = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.id} {self.input_text} {self.text_length}"
