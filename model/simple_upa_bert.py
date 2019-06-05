import torch
import logging
from pytorch_pretrained_bert.modeling import BertModel
from model.components.user_product_attention import UserProductAttention

log_format = '%(asctime)-10s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)


class SimpleUPABert(torch.nn.Module):

    def __init__(self, n_user, n_product, n_classes,
                 user_size=200, product_size=200, attention_hidden_size=200, hidden_size=768):
        super(SimpleUPABert, self).__init__()
        self.n_user = n_user
        self.n_product = n_product
        self.n_classes = n_classes
        self.user_size = user_size
        self.product_size = product_size
        self.hidden_size = hidden_size
        self.Uemb = torch.nn.Embedding(n_user, user_size)
        self.Pemb = torch.nn.Embedding(n_product, product_size)
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.word_attention = UserProductAttention(
            user_size, product_size, attention_hidden_size, hidden_size)
        self.linear = torch.nn.Linear(self.hidden_size, self.n_classes)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, batch):
        """
            user_ids [batch_size]
            product_ids [batch_size]
            input_ids [batch_size, 512]
            input_mask [batch_size, 512]

        """
        user_ids, product_ids, _, input_ids, input_mask = batch
        user_embs = self.Uemb(user_ids)
        product_embs = self.Pemb(product_ids)
        bert_out, _ = self.bert(
            input_ids, attention_mask=input_mask, output_all_encoded_layers=False)
        word_attention_out = self.word_attention(
            bert_out, user_embs, product_embs, None)
        linear_out = self.linear(word_attention_out)
        softmax_out = self.softmax(linear_out)
        return softmax_out

    def train(self):
        self.bert.train()

    def eval(self):
        self.bert.eval()

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.word_attention = self.word_attention.to(*args, **kwargs)
        return self
