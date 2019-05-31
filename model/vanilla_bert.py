import torch
import logging
from pytorch_pretrained_bert.modeling import BertModel

log_format = '%(asctime)-10s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)


class VanillaBert(torch.nn.Module):

    def __init__(self, n_classes, hidden_size=768):
        super(VanillaBert, self).__init__()
        self.n_classes = n_classes
        self.hidden_size = hidden_size
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.linear = torch.nn.Linear(self.hidden_size, self.n_classes)
        self.softmax = torch.nn.Softmax()

    def forward(self, batch):
        _, _, _, input_ids, input_mask = batch
        bert_out, _ = self.bert(
            input_ids, attention_mask=input_mask, output_all_encoded_layers=False)
        linear_out = self.linear(bert_out[:, 0, :])
        softmax_out = self.softmax(linear_out)
        return softmax_out

    def train(self):
        # this is called in the pytorch example, I dont't know why
        self.bert.train()

    def eval(self):
        self.bert.eval()
