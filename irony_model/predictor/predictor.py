
from typing import Tuple

from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor



@Predictor.register('ironic-predictor')
class IronicTweetPredictor(Predictor):
    """Predictor Wrapper for Irony Classification Task"""
    @overrides
    def _json_to_instance(self, jsonDict : JsonDict) -> Tuple[Instance, JsonDict]:
        tweet = jsonDict['tweet']
        instance = self._dataset_reader.text_to_instance(input = tweet)

        label_dict = self._model.vocab.get_index_to_token_vocabulary('labels')

        all_labels = [label_dict[i] for i in range(len(label_dict))]

        return instance
    