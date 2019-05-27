import torch
import logging
from user_product_attention import UserProductAttention

log_format = '%(asctime)-10s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)

class VanillaUPA(torch.nn.Module):

    def __init__(self, n_user, n_product, n_classes, n_words, user_size=200, product_size=200, word_size=200, hidden_size=200):
        super(VanillaUPA, self).__init__()
        self.Uemb = torch.nn.Embedding(n_user, user_size)
        self.Pemb = torch.nn.Embedding(n_product, product_size)
        self.Wemb = torch.nn.Embedding(n_words, word_size)
        self.lstm1 = torch.nn.LSTM(word_size, word_size)
        self.word_attention = UserProductAttention()
        self.lstm2 = torch.nn.LSTM(hidden_size, hidden_size)
        self.sentence_attention = UserProductAttention()
        self.linear = torch.nn.Linear(hidden_size, n_classes)
        self.softmax = torch.nn.Softmax()
    def forward(self, input_ids, input_mask, user_ids, product_ids, sentence_offsets):
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
        word_embs = self.Wemb(input_ids)
        lstm1_out, _ = self.lstm1(word_embs)
        word_attention_out = self.word_attention(lstm1_out, user_embs, product_embs, sentence_offsets)
        lstm2_out, _ = self.lstm2(word_attention_out)
        sentence_attention_out = self.sentence_attention(lstm2_out, user_embs, product_embs)
        linear_out = self.linear(sentence_attention_out)
        softmax_out = self.softmax(linear_out)
        return softmax_out
