import os
import inspect
import pickle

app_path = inspect.getfile(inspect.currentframe())
sub_dir = os.path.realpath(os.path.dirname(app_path))
main_dir = os.path.dirname(sub_dir)

from django.apps import AppConfig
import tensorflow as tf


class DetectorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'detector'

    lstm_model = tf.keras.models.load_model(os.path.join(main_dir, 'trained_models/detection_model.h5'))

    with open(os.path.join(main_dir, 'trained_models/tokenizer_mdl.pkl'), 'rb') as f:
        token_model = pickle.load(f)

