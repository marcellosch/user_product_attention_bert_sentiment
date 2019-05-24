import torch
import logging
from pytorch_pretrained_bert.modeling import BertModel

log_format = '%(asctime)-10s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)


class UserProductBert(torch.nn.Module):

    def __init__(self, n_user, n_product, n_classes, user_size=200, product_size=200, hidden_size=768):
        self.Uemb = torch.nn.Embedding(n_user, user_size)
        self.Pemb = torch.nn.Embedding(n_product, product_size)
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.word_attention = UserProductAttention()
        self.lstm = torch.nn.LSTM(hidden_size, hidden_size)
        self.sentence_attention = UserProductAttention()
        self.linear = torch.nn.Linear(self.hidden_size, self.n_classes)
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
        bert_out, _ = self.bert(input_ids, output_all_encoded_layers=False)
        word_attention_out = self.word_attention(bert_out, user_embs, product_embs, sentence_offsets)
        lstm_out, _ = self.lstm(word_attention_out)
        sentence_attention_out = self.sentence_attention(lstm_out, user_embs, product_embs) 
        linear_out = self.linear(sentence_attention_out)
        softmax_out = self.softmax(linear_out)
        return softmax_out

    def train():
        # this is called in the pytorch example, I dont't know why
        self.bert.train()

class SimpleUserProductBert(torch.nn.Module):

    def __init__(self, n_user, n_product, n_classes, user_size=200, product_size=200, hidden_size=768):
        super(SimpleUserProductBert, self).__init__()
        self.user_size = user_size
        self.product_size = product_size
        self.hidden_size = hidden_size
        self.Uemb = torch.nn.Embedding(n_user, user_size)
        self.Pemb = torch.nn.Embedding(n_product, product_size)
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.word_attention = UserProductAttention()
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

class UserProductAttention(torch.nn.Module):
  def __init__(self, user_size=200, product_size=200, out_size=200, hidden_size=768):
    super(UserProductAttention, self).__init__()
    self.user_size = user_size
    self.product_size = product_size
    self.hidden_size = hidden_size
    self.out_size = out_size
    self.Wh = torch.nn.Linear(hidden_size, out_size, bias=False)
    self.Wu = torch.nn.Linear(user_size, out_size, bias=False)
    self.Wp = torch.nn.Linear(product_size, out_size, bias=False)
    self.v = torch.nn.Linear(out_size, 1, bias=False)
    self.b = torch.zeros(1, 1,out_size)
    self.tanh = torch.nn.Tanh()
    self.softmax = torch.nn.Softmax(dim=1)
    
  def forward(self, H, u, p, sentence_offsets=None):
    """expect H in [batch_size, seq_len, hidden_size]
              u in [batch_size, user_size]
              p in [batch_size, product_size] 
    """
    batch_size, seq_len, _ = H.shape
    ut = self.Wu(u) # [batch_size,out_size]
    ut = ut.unsqueeze(1).repeat(1,seq_len,1) #                 [batch_size, seq_len, out_size]
    pt = self.Wp(p).unsqueeze(1).repeat(1,seq_len,1) #       [batch_size, seq_len, out_size]
    bt = self.b.repeat(batch_size,seq_len,1) #  [batch_size, seq_len, out_size]
    Ht = self.Wh(H) #                           [batch_size, seq_len, out_size]
    alphas = None
    if sentence_offsets is None: # all inputs belong to the same sequence
        raw_alphas = self.v(self.tanh(Ht + ut + pt + bt)) # [batch_size, seq_len, 1]
        alphas = self.softmax(raw_alphas)
        return H.transpose(1,2).matmul(alphas).squeeze() # [batch_size, hidden_size]
    else: # inputs have to be softmaxed in groups defined by sentence_offsets
        # TODO: implement softmax relative to offsets
        out = torch.Tensor(seq_len, )
        pass

