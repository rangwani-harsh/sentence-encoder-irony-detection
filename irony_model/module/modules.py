from torch import nn, torch
from torch.autograd import Variable
import torch.nn.functional as F

from typing import List
from overrides import overrides

from allennlp.modules import Seq2SeqEncoder, Seq2VecEncoder
from allennlp.data.tokenizers.word_splitter import WordSplitter
from allennlp.data.tokenizers.token import Token

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons


@Seq2VecEncoder.register('basic_self_attention')
class SelfAttention(Seq2VecEncoder):
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
        loss = 0 # In case of some regularization penality required 
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

        return representations, scores, loss


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
                 emoticon_unpack = False,
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

@Seq2VecEncoder.register('structured-self-attention')
class StructuredSelfAttention(Seq2VecEncoder):
    """ A seq2seq encoder which calculates the sentence representation
    using the output states of an LSTM. This implements the paper arXiv:1703.03130. 

    Input : [batch_size, num_rows, hidden_dim]
    Output:  [batch_size, hidden_dim] # Final representation of the sentence
    """

    def __init__(self, 
                attention_size, #Attention layer size
                attention_heads, #Number of attention heads
                hidden_dims, # Hidden Dimension of the LSTM encoder
                regularization = False, #Forbenius Norm Based Regularization method
                penalty_coeffecient = 0.01 #Coeffoecient of regularization
                ):
            """
            Initializes parameters of the model

            attention_size : {int} The attention layer size i.e. d_a in paper
            attention_heads : {int} The number of different attention heads in sentence i.e. r in paper
            hidden_dims : {int} The hidden dimensions of the lstm upon which the attention operates
            """
            super(StructuredSelfAttention, self).__init__()
            self.attention_heads = attention_heads
            self.attention_size = attention_size
            self.regularization = regularization
            self.penalty_coeffecient = penalty_coeffecient
            self.linear_first = torch.nn.Linear(hidden_dims, attention_size, bias = False) 
            self.linear_second = torch.nn.Linear(attention_size, attention_heads, bias = False)
            self.init_weights()

    def init_weights( self ):
        initrange = 0.1
        self.linear_first.weight.data.uniform_( -initrange, initrange )
        self.linear_second.weight.data.uniform_( -initrange, initrange )

    def softmax(self,input, axis=1):
        """
        Softmax applied to axis=n
 
        Args:
           input: {Tensor,Variable} input on which softmax is to be applied
           axis : {int} axis on which softmax is to be applied
 
        Returns:
            softmaxed tensors
 
        #TODO Rewrite the function again if time permits   
        """
 
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size)-1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = F.softmax(input_2d)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size)-1)

    def forward(self, inputs, mask):

        """Gives out the attention, sentence representation and the regularization loss 
           Args:
            input: {torch.Tensor} input of the underlying seq2seq encoder
            mask:  {torch.tensor} contains mask of length of sentence.
           Returns:
            sentence_representation: {Torch.tensor}
            loss: {Torch.tensor} Regularization Penalty
            attention: {Torch.tensor} Attention over the different parts of the sentence
        """
        loss = 0
        transformed_inputs = torch.tanh(self.linear_first(inputs)) # n * d_a
        transformed_inputs = self.linear_second(transformed_inputs) # n * r


        attention = self.softmax(transformed_inputs, axis = 1)*mask.unsqueeze(2) # n*r
        attention = attention.transpose(1,2) # n * r => r*n
        _sums = torch.sum(attention, dim=2, keepdim=True) #Apply mask and then normalize
        attention = attention.div(_sums)

        batch_size = inputs.shape[0] #As the batch_size if first dimension #TODO Find a better way

        if self.regularization:
            attentionT = attention.transpose(1, 2)
            identity = torch.eye(attention.size(1))
            identity = Variable(identity.unsqueeze(0).expand(batch_size,attention.size(1),attention.size(1)).cuda())
            penal = self.l2_matrix_norm(attention@attentionT - identity)
            loss += (self.penalty_coeffecient * penal/batch_size).type(torch.FloatTensor)

        sentence_embeddings = attention@inputs #Matrix Multiplication r*n*n*hidden_dim => r*hidden_dims
        avg_sentence_embeddings = (torch.sum(sentence_embeddings,1)/self.attention_heads).squeeze() #r*hidden_dims => hidden_dims 

        #Average attention too (Makes it easy for visulization)
        attention  = (torch.sum(attention, 1)/self.attention_heads).squeeze()

        return avg_sentence_embeddings, attention, loss



    #L2 Regularized Norm
    def l2_matrix_norm(self,m):
        """
        Frobenius norm calculation
 
        Args:
           m: {Variable} ||AAT - I||
 
        Returns:
            regularized value
 
       
        """
        return torch.sum(torch.sum(torch.sum(m**2,1),1)**0.5).type(torch.DoubleTensor)

