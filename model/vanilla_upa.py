import torch
import logging
from .components.user_product_attention import UserProductAttention
from .components.bert_word_embeddings import BertWordEmbeddings

log_format = '%(asctime)-10s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)


class VanillaUPA(torch.nn.Module):

    def __init__(self, n_user, n_product, n_classes, user_size=200, product_size=200, hidden_size=200):
        super(VanillaUPA, self).__init__()
        self.n_user = n_user
        self.n_product = n_product
        self.n_classes = n_classes
        self.user_size = user_size
        self.product_size = product_size
        self.hidden_size = hidden_size
        self.Uemb = torch.nn.Embedding(n_user, user_size)
        self.Pemb = torch.nn.Embedding(n_product, product_size)
        self.Wemb = BertWordEmbeddings.from_pretrained('bert-base-uncased')
        self.word_size = self.Wemb.word_embeddings.embedding_dim
        self.lstm1 = torch.nn.LSTM(self.word_size, hidden_size)
        self.word_attention = UserProductAttention(
            self.user_size, self.product_size, 200, self.hidden_size)
        self.lstm2 = torch.nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.sentence_attention = UserProductAttention(
            self.user_size, self.product_size, 200, self.hidden_size)
        self.linear = torch.nn.Linear(hidden_size, n_classes)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, batch):
        user_ids, product_ids, _, sentence_matrix = batch
        assert sentence_matrix.shape[0]%user_ids.shape[0] == 0
        max_sentence_count = sentence_matrix.shape[0] // user_ids.shape[0]
        user_embs = self.Uemb(user_ids)
        product_embs = self.Pemb(product_ids)
        word_embs = self.Wemb(sentence_matrix)
        lstm1_out, _ = self.lstm1(word_embs)
        repeated_user_embs = user_embs.repeat_interleave(
            max_sentence_count, dim=0)
        repeated_product_embs = product_embs.repeat_interleave(
            
            max_sentence_count, dim=0)
        word_attention_out = self.word_attention(
            lstm1_out, repeated_user_embs, repeated_product_embs)
        word_attention_out = word_attention_out.view(
            -1, max_sentence_count, self.hidden_size)
        lstm2_out, _ = self.lstm2(word_attention_out)
        sentence_attention_out = self.sentence_attention(
            lstm2_out, user_embs, product_embs)
        linear_out = self.linear(sentence_attention_out)
        softmax_out = self.softmax(linear_out.view(-1, self.n_classes))
        return softmax_out

    def train(self):
        pass

    def eval(self):
        pass

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.word_attention = self.word_attention.to(*args, **kwargs)
        self.sentence_attention = self.sentence_attention.to(*args, **kwargs)

        return self
