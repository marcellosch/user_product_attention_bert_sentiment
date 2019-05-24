import torch
import logging

log_format = '%(asctime)-10s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)


class UserProductBert(torch.nn.Module):

    def __init__(self, n_user, n_product, user_size=200, product_size=200, hidden_size=768):
        self.Uemb = torch.nn.Embedding(n_user, user_size)
        self.Pemb = torch.nn.Embedding(n_product, product_size)
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.attention0 = UserProductAttention()
        self.lstm = torch.nn.LSTM(hidden_size, hidden_size)
        self.attention1 = UserProductAttention()
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
        attention0_out = self.attention0(bert_out, user_embs, product_embs, sentence_offsets)
        lstm_out, _ = self.lstm(attention0_out)
        attention1_out = self.attention1(lstm_out, user_embs, product_embs) 
        linear_out = self.linear(attention1_out)
        softmax_out = self.softmax(linear_out)
        return softmax_out


        
    

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

    self.v = torch.nn.Linear(out_size, 1, biase=False)
    self.b = torch.zeros(1,out_size)
    self.tanh = torch.nn.Tanh()
    self.softmax = torch.nn.Softmax(dim=0)
    
  def forward(self, H, u, p, sentence_offsets =None):
    """expect H in seq_len x hidden_size
              u in user_size x 1
              p in product_size x 1
    """
    seq_len=H.shape[0]
    ut = self.Wu(u).repeat(seq_len,1)
    pt = self.Wp(p).repeat(seq_len,1)
    bt = self.b.repeat(seq_len,1)
    Ht = self.Wh(H)
    alphas = None
    if sentence_offsets None: # all inputs belong to the same sequence
        alphas = self.softmax(self.v(self.tanh(Ht + ut + pt + bt)))
        return alphas.transpose(1,0).matmul(H)
    else: # inputs have to be softmaxed in groups defined by sentence_offsets
        # TODO: implement softmax relative to offsets
        pass

