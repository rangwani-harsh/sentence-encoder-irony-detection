from torch import nn, torch
from torch.autograd import Variable

from typing import List
from overrides import overrides

from allennlp.modules import Seq2SeqEncoder
from allennlp.data.tokenizers.word_splitter import WordSplitter
from allennlp.data.tokenizers.token import Token

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons


@Seq2SeqEncoder.register('basic_self_attention')
class SelfAttention(Seq2SeqEncoder):
    """A simple Seq2Seq encoder that calculates attention over
    all the outputs of time step t and gives a final representation.
    It's just the weighted output of each time step as the final representation.

    Input : [batch_size, num_rows, hidden_dim]
    Output : [batch_size, hidden_dim]
    """

    def __init__(self, attention_size,
                 batch_first=False,
                 layers=1,
                 dropout=.0,
                 non_linearity="tanh"):
        super(SelfAttention, self).__init__()

        self.batch_first = batch_first

        if non_linearity == "relu":
            activation = nn.ReLU()
        else:
            activation = nn.Tanh()

        modules = []
        for i in range(layers - 1):
            modules.append(nn.Linear(attention_size, attention_size))
            modules.append(activation)
            modules.append(nn.Dropout(dropout))

        # last attention layer must output 1
        modules.append(nn.Linear(attention_size, 1))
        modules.append(activation)
        modules.append(nn.Dropout(dropout))

        self.attention = nn.Sequential(*modules)

        self.softmax = nn.Softmax(dim=-1)



        ##################################################################
        # STEP 1 - perform dot product
    def forward(self, inputs, mask):
        # of the attention vector and each hidden state
        ##################################################################

        # inputs is a 3D Tensor: batch, len, hidden_size
        # scores is a 2D Tensor: batch, len
        scores = self.attention(inputs).squeeze()
        scores = self.softmax(scores)

        ##################################################################
        # Step 2 - Masking
        ##################################################################

        # apply the mask - zero out masked timesteps
        masked_scores = scores * mask

        # re-normalize the masked scores
        _sums = masked_scores.sum(-1, keepdim=True)  # sums per row
        scores = masked_scores.div(_sums)  # divide by row sum

        ##################################################################
        # Step 3 - Weighted sum of hidden states, by the attention scores
        ##################################################################

        # multiply each hidden state with the attention weights
        weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs))

        # sum the hidden states
        representations = weighted.sum(1).squeeze()

        return representations, scores


@WordSplitter.register('twitter')
class TwitterWordSplitter(WordSplitter):
    """
        Word splitter based on ekphrasis for
        downstream tasks on tweets.
        Uses ekhphrasis module for the purpose.
    """
    def __init__(self,
                 normalize = ['url', 'email', 'percent', 'money', 
                    'phone', 'user','time','date', 'number'],
                 annotate = [],
                 all_caps_tag = "wrap",
                 fix_text = True,
                 segmenter = "twitter_2018",
                 corrector="twitter_2018",
                 unpack_hashtags=True,
                 unpack_contractions=True,
                 spell_correct_elong=False,
                 emoticon_unpack = True,
                 lowercase = False
                ):

            self.emoticons = emoticon_unpack
            if self.emoticons:
                emoticon_list = [emoticons]
            else:
                emoticon_list = []
            
            self.tweet_splitter = TextPreProcessor(
                normalize= normalize,
                annotate=annotate,
                all_caps_tag=all_caps_tag,
                fix_text=fix_text,
                segmenter=segmenter,
                corrector=corrector,
                unpack_hashtags=unpack_hashtags,
                unpack_contractions=unpack_contractions,
                spell_correct_elong=False,
                tokenizer=SocialTokenizer(lowercase=lowercase).tokenize,
                dicts=emoticon_list
            )

    @overrides
    def split_words(self, sentence: str) -> List[Token]:
        # This works because our Token class matches spacy's.
        tokens = [] # As allennlp requires the datatype Tokens
        for text in self.tweet_splitter.pre_process_doc(sentence):
                tokens.append(Token(text))
        return tokens

                