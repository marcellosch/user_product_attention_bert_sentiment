import torch
import logging
from pytorch_pretrained_bert.modeling import BertModel
from user_product_attention import UserProductAttention

log_format = '%(asctime)-10s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)

class SimpleUserProductBert(torch.nn.Module):

    def __init__(self, n_user, n_product, n_classes, 
            user_size=200, product_size=200, attention_hidden_size=200, hidden_size=768):
        super(SimpleUserProductBert, self).__init__()
        self.n_user = n_user
        self.n_product = n_product
        self.n_classes = n_classes
        self.user_size = user_size
        self.product_size = product_size
        self.hidden_size = hidden_size
        self.Uemb = torch.nn.Embedding(n_user, user_size)
        self.Pemb = torch.nn.Embedding(n_product, product_size)
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.word_attention = UserProductAttention(user_size, product_size, attention_hidden_size, hidden_size)
        self.linear = torch.nn.Linear(self.hidden_size, self.n_classes)
        self.softmax = torch.nn.Softmax()
    def forward(self, input_ids, input_mask, user_ids, product_ids, sentence_offsets=None):
        """ Inputs:
            `input_ids`: Tensor of shape [batch_size, max_seq_length] containing one documents word ids per row
            `input_mask`: torch.LongTensor of shape [batch_size, max_seq_length] with indices in [0,1]. The mask is used to mask sentences that are shorter than max_seq_length
            list of dicts: input_ids, input mask, token_type_ids, user_id, product_id, sentence_offsets
            `user_ids`: torch.LongTensor of shape [batch_size] that denotes the user id for documents
            `product_ids`: torch.LongTensor of shape [batch_size] that denotes the product ids for documents
            `sentence_offsets`: list of batch_size iterables where each contains the integer offsets of the sentence starts in document 
        """
        user_embs = self.Uemb(user_ids)
        product_embs = self.Pemb(product_ids)
        bert_out, _ = self.bert(input_ids, output_all_encoded_layers=False)
        word_attention_out = self.word_attention(bert_out, user_embs, product_embs, sentence_offsets)
        linear_out = self.linear(word_attention_out)
        softmax_out = self.softmax(linear_out)
        return softmax_out

    def train():
        # this is called in the pytorch example, I dont't know why
        self.bert.train()


