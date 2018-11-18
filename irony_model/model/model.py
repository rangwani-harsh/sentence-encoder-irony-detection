from typing import Dict, Optional, List

import numpy
from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder,Seq2SeqEncoder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy, F1Measure


@Model.register("irony_classifier")
class IronyTweetClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 tweet_encoder: Seq2SeqEncoder,
                 class_weights: torch.LongTensor,
                 classifier_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(IronyTweetClassifier, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.tweet_encoder = tweet_encoder
        self.classifier_feedforward = classifier_feedforward
        self.metrics = {
                "accuracy": CategoricalAccuracy()
        }
        self._unlabelled_f1 = F1Measure(positive_label=1)
        if text_field_embedder.get_output_dim() != tweet_encoder.get_input_dim():
            raise ConfigurationError("The output dimension of the text_field_embedder must match the "
                                     "input dimension of the title_encoder. Found {} and {}, "
                                     "respectively.".format(text_field_embedder.get_output_dim(),
                                                            tweet_encoder.get_input_dim()))
        self.loss = torch.nn.CrossEntropyLoss()
        self.loss_multiclass = torch.nn.CrossEntropyLoss(weight=torch.Tensor(class_weights))
        self.single_task = torch.nn.Linear(in_features= classifier_feedforward.get_output_dim(), out_features = 2)
        self.multiclass_task = torch.nn.Linear(in_features= classifier_feedforward.get_output_dim(), out_features = 4)
        initializer(self)



    def forward(self,
                tweet: Dict[str, torch.LongTensor],
                sentence_embeds: torch.LongTensor,
                label: torch.LongTensor = None,
                multiclass_label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:

        embedded_tweet = self.text_field_embedder(tweet) # This will take the tweet and initialize into embedding (i.e char and token) (If we keep only our title as tokens in the tweet instance the dict['tokens'] will map to token ids i.e shape(#TODO))
        tweet_mask = util.get_text_field_mask(tweet).float()
        encoded_tweet = self.tweet_encoder(embedded_tweet, tweet_mask) # An LSTM or any other seq encode

        encoded_tweet_last = encoded_tweet[:, -1, :] #Flatten out (batch_size, time_steps, dimensions) to (batch_size, time_steps * dimensions)
        encoded_tweet_first = encoded_tweet[:, 0, :]
        encoded_tweet = torch.cat([encoded_tweet_first, encoded_tweet_last], dim = 1)

        concatenated_representatation = torch.cat([sentence_embeds, encoded_tweet], dim = -1)

        logits = self.classifier_feedforward(concatenated_representatation)
        singleclass_logits = self.single_task(logits)
        multiclass_logits = self.multiclass_task(logits)
        class_probabilities_single = F.softmax(singleclass_logits)
        class_probabilities_multi  = F.softmax(multiclass_logits)

        output_dict = {"class_probabilities": class_probabilities_single, "class_probablities_multi": class_probabilities_multi}

        #To make it run in case of demo
        if label is not None:

            loss = self.loss(singleclass_logits, label)
            for metric in self.metrics.values():
                metric(singleclass_logits, label)
            self._unlabelled_f1(singleclass_logits, label)
            output_dict["loss"] = loss 

        if multiclass_label is not None:
            loss_multiclass = self.loss_multiclass(multiclass_logits,multiclass_label)

            if "loss" not in output_dict:
                output_dict["loss"] = 0

            output_dict["loss"] += loss_multiclass

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}
        precision, recall, f1_measure = self._unlabelled_f1.get_metric(reset)
        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1"] = f1_measure
        return metrics

    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        predictions = output_dict['class_probabilities'].cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels")
                  for x in argmax_indices]
        output_dict['label'] = labels
        return output_dict
