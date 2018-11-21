
from typing import Tuple

from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor

import numpy as np

TEST_EMBEDDINGS_PATH = "dataset/encoding_test.npy" #TODO Change it to path from config of the model
TEST_EMBEDDINGS = np.load(TEST_EMBEDDINGS_PATH) #TODO Load it only once


@Predictor.register('ironic-predictor')
class IronicTweetPredictor(Predictor):
    """Predictor Wrapper for Irony Classification Task"""

    @overrides
    def _json_to_instance(self, jsonDict : JsonDict) -> Tuple[Instance, JsonDict]:
        tweet = jsonDict['tweet']

        tweet_index_dataset = jsonDict['index']

        #Load test embeddings of deepmoji
        instance = self._dataset_reader.text_to_instance(input = tweet, embedding = TEST_EMBEDDINGS[tweet_index_dataset])

        label_dict = self._model.vocab.get_index_to_token_vocabulary('labels')

        all_labels = [label_dict[i] for i in range(len(label_dict))]

        return instance
    