# pylint: disable=no-self-use,invalid-name,unused-import
from unittest import TestCase

from pytest import approx
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

import irony_model

class TestIronyModelPredictor(TestCase):
    def test_tweet_prediction_task(self):
        input_tweet = {
            'tweet' : 'What a hell of a night! Sleeping all day!!'
        }

        archived_model = load_archive("~/tmp/models/irony/model.tar.gz")
        predictor = Predictor.from_archive(archived_model, "irony_classifier")
        
        result = predictor.predict_json(input_tweet)
        label = result.get("label")
        
        class_probabilities = result.get("class_probabilities")
        assert class_probabilities is not None
        assert all(cp > 0 for cp in class_probabilities)
        assert sum(class_probabilities) == approx(1.0)


