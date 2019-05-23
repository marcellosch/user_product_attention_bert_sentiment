import torch


class UserProductBert(torch.nn.Module):

    def __init__(self):
        self.bert = None

    def forward(self, inputs):
        h = self.bert(inputs)
        pass
    