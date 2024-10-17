import os
import inspect
import logging
import traceback  # noqa

app_path = inspect.getfile(inspect.currentframe())
sub_dir = os.path.realpath(os.path.dirname(app_path))
main_dir = os.path.dirname(sub_dir)

from django.shortcuts import render
from django.views import View
from tensorflow.keras.preprocessing.sequence import pad_sequences

from .apps import DetectorConfig
from .models import News

NUMBER_WORDS = 100

if not os.path.exists('logs'):
    os.mkdir('logs')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.StreamHandler()
file_handler = logging.FileHandler('logs/api_log.log')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


class FakeNewsView(View):
    template_name = 'fake_news.html'

    def get(self, request):
        return render(request, self.template_name)

    def post(self, request):
        threshold = 0.5
        input_text: str = request.POST.get('input_text', '')
        tokenizer = DetectorConfig.token_model
        lstm_model = DetectorConfig.lstm_model
        text_seq = tokenizer.texts_to_sequences([input_text])
        test_input = pad_sequences(text_seq, maxlen=NUMBER_WORDS)

        output = lstm_model.predict(test_input)

        if output >= threshold:
            context = {
                'input_text': input_text,
                'is_genuine': 1,
                'probability': round(output[0][0] * 100, 2)
            }
            news_object = News(input_text=input_text, text_length=len(input_text), genuine=True)
            news_object.save()
        else:
            context = {
                'input_text': input_text
            }
            news_object = News(input_text=input_text, text_length=len(input_text))
            news_object.save()

        return render(request, self.template_name, context)
