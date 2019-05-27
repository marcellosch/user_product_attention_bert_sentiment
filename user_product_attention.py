import torch
import logging

log_format = '%(asctime)-10s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)

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
    
  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs) 
    self.b = self.b.to(*args, **kwargs) 
    return self

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
    print("H {}".format(H.device))
    print("u {}".format(u.device))
    print("p {}".format(p.device))
    print("self.Wh {}".format(self.Wh.device))
    print("self.Wu {}".format(self.Wu.device))
    print("self.Wp {}".format(self.Wp.device))
    print("self.v {}".format(self.v.device))
    print("self.b {}".format(self.Wh.device))
    alphas = None
    if sentence_offsets is None: # all inputs belong to the same sequence
        raw_alphas = self.v(self.tanh(Ht + ut + pt + bt)) # [batch_size, seq_len, 1]
        alphas = self.softmax(raw_alphas)
        return H.transpose(1,2).matmul(alphas).squeeze() # [batch_size, hidden_size]
    else: # inputs have to be softmaxed in groups defined by sentence_offsets
        # TODO: implement softmax relative to offsets
        out = torch.Tensor(seq_len, )
        pass



