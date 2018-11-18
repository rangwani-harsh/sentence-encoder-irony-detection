from typing import Dict
import json
import logging

from overrides import overrides

import tqdm
import numpy as np
import csv

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class FieldPreparator():
    def __init__(self,
                 field: str = None,
                 mapping: Dict = {}):
        self._field = field
        self._mapping = mapping

    def transform(self, field, value) -> str:
        if field == self._field:
            return self._mapping.get(value, default=value)
        else:
            return value

    @classmethod
    def from_params(cls, params: Params) -> 'FieldPreparator':
        field = params.pop('field', None)
        mapping = params.pop('mapping', {}).as_dict()
        return FieldPreparator(field=field, mapping=mapping)



@DatasetReader.register("csv_classification_reader")
class CsvClassificationReader(DatasetReader):
    """
    Reads a file from a classification dataset.  This data is
    formatted as jsonl, one json-formatted instance per line.  The positions in the CSV file can defined in
    the JSON definition.
    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
         See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        See :class:`TokenIndexer`.
    """

    def __init__(self,
                 pos_input: int,
                 pos_gold_label: int,
                 skip_header: bool = True,
                 delimiter: str = ",",
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        self._tokenizer = tokenizer or WordTokenizer()
        self._input = pos_input
        self._gold_label = pos_gold_label
        self._skip_header = skip_header
        self._delimiter = delimiter
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    
    def parse_dataset(self, fp):
        '''
        Loads the dataset .txt file with label-tweet on each line and parses the dataset.
        :param fp: filepath of dataset
        :return:
            corpus: list of tweet strings of each tweet.
            y: list of labels
        '''
        y = []
        corpus = []
        with open(fp, 'rt') as data_in:
            for line in data_in:
                if not line.lower().startswith("tweet index"): # discard first line if it contains metadata
                    line = line.rstrip() # remove trailing whitespace
                    label = (line.split("\t")[1])
                    tweet = line.split("\t")[2]
                    y.append(label)
                    corpus.append(tweet)

        return corpus, y

    @overrides
    def read(self, file_path: list):
        # if `file_path` is a URL, redirect to the cache
        taskA_file_path = file_path['taskA']
        taskB_file_path = file_path['taskB']
        emb_file_path = file_path['emb']
        
        instances = []
        
        logger.info("Reading instances from CSV dataset at: %s", file_path)

        embedding = np.load(emb_file_path)
        corpus, labels = self.parse_dataset(taskA_file_path)
        _, multiclass_labels = self.parse_dataset(taskB_file_path)
        
        assert len(labels) == len(multiclass_labels)
        logger.info("Processing the number of examples {}".format(len(labels)))

        # examples = [make_example(line, fields) for line in reader]
        for i in range(0, len(labels)):
            input = corpus[i]
            label = (labels[i])
            multiclass_label = multiclass_labels[i]
            instances.append(self.text_to_instance(input, embedding[i], label, multiclass_label))

        if not instances:
            raise ConfigurationError("No instances were read from the given filepath {}. "
                                     "Is the path correct?".format(file_path))
        return instances

    @overrides
    def text_to_instance(self,  # type: ignore
                         input: str,
                         embedding : np.array,
                         label: str = None,
                         multiclass_label: str = None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        input_tokens = self._tokenizer.tokenize(input)
        fields['tweet'] = TextField(input_tokens, self._token_indexers)
        fields['sentence_embeds'] = ArrayField(embedding)
        if label:
            fields['label'] = LabelField(label)
        if multiclass_label:
            fields['multiclass_label'] = LabelField(multiclass_label)
        return Instance(fields)

    # @classmethod
    # def from_params(cls, params: Params) -> 'CsvClassificationReader':
    #     tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
    #     input = params.pop('pos_input', None)
    #     gold_label = params.pop('pos_gold_label', None)
    #     skip_header = params.pop('skip_header', True)
    #     delimiter = params.pop('delimiter', None)
    #     token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', None))
    #     params.assert_empty(cls.__name__)
    #     return CsvClassificationReader(tokenizer=tokenizer,
    #                                    token_indexers=token_indexers,
    #                                    skip_header=skip_header,
    #                                    delimiter=delimiter,
    #                                    input=input,
    #                                    gold_label=gold_label)