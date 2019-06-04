import torch
import logging
from pytorch_pretrained_bert.modeling import BertModel
from model.components.user_product_attention import UserProductAttention

log_format = '%(asctime)-10s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)


class UPABert(torch.nn.Module):

    def __init__(self, n_user, n_product, n_classes, user_size=200, product_size=200, attention_hidden_size=200, hidden_size=768):
        super(UPABert, self).__init__()
        self.n_user = n_user
        self.n_product = n_product
        self.n_classes = n_classes
        self.user_size = user_size
        self.product_size = product_size
        self.attention_hidden_size = attention_hidden_size
        self.hidden_size = hidden_size

        self.Uemb = torch.nn.Embedding(n_user, user_size)
        self.Pemb = torch.nn.Embedding(n_product, product_size)
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.word_attention = UserProductAttention(user_size, product_size, attention_hidden_size, hidden_size)
        self.lstm = torch.nn.LSTM(hidden_size, hidden_size)
        self.sentence_attention = UserProductAttention()
        self.linear = torch.nn.Linear(self.hidden_size, self.n_classes)
        self.softmax = torch.nn.LogSoftmax()

    def forward(self, batch):
        """ Inputs:
            `input_ids`: Tensor of shape [batch_size, max_seq_length] containing one documents word ids per row
            `input_mask`: torch.LongTensor of shape [batch_size, max_seq_length] with indices in [0,1]. The mask is used to mask sentences that are shorter than max_seq_length
            list of dicts: input_ids, input mask, token_type_ids, user_id, product_id, sentence_offsets
            `user_ids`: torch.LongTensor of shape [batch_size] that denotes the user id for documents
            `product_ids`: torch.LongTensor of shape [batch_size] that denotes the product ids for documents
        """
        user_ids, product_ids, _, sentence_matrix = batch
        max_sentence_count = sentence_matrix.shape[0] //user_ids.shape[0]
        user_embs = self.Uemb(user_ids)
        product_embs = self.Pemb(product_ids)
        bert_out, _ = self.bert(sentence_matrix, output_all_encoded_layers=False)
        repeated_user_embs = user_embs.repeat_interleave(max_sentence_count, dim=0)
        repeated_product_embs = product_embs.repeat_interleave(max_sentence_count, dim=0)
        word_attention_out = self.word_attention(
            bert_out, repeated_user_embs, repeated_product_embs)
        word_attention_out = word_attention_out.view(-1, max_sentence_count, self.hidden_size)
        lstm_out, _ = self.lstm(word_attention_out)
        sentence_attention_out = self.sentence_attention(
            lstm_out, user_embs, product_embs)
        linear_out = self.linear(sentence_attention_out)
        softmax_out = self.softmax(linear_out)
        return softmax_out

    def train(self):
        self.bert.train()

    def eval(self):
        self.bert.eval()
