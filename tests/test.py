#! /usr/bin/python3

import unittest
from model import *
from model.upa_bert import *
from model.vanilla_bert import *
from model.vanilla_upa import *


class TestModel(unittest.TestCase):
    def test_UserProductAttention_dimensions(self):
        upa = UserProductAttention()
        H = torch.rand(64, 512, 768)
        u = torch.rand(64, 200)
        p = torch.rand(64, 200)
        out = upa(H, u, p)
        self.assertEqual(out.shape, (64, 768))

    def test_SimpleUPABert(self):
        supb = SimpleUPABert(n_user=100, n_product=200, n_classes=5)
        input_ids = (torch.rand(2, 512)*800).long()
        input_mask = torch.ones(2, 512).long()
        user_ids = (torch.rand(2) * 100).long()
        product_ids = (torch.rand(2)*200).long()
        batch = (user_ids, product_ids, None,
                 input_ids, None, input_mask, None, None)
        out = supb(batch)
        self.assertEqual(out.shape, (2, 5))

    def test_VanillaBert(self):
        vbert = VanillaBert(n_classes=5)
        input_ids = (torch.rand(2, 512)*800).long()
        input_mask = torch.ones(2, 512).long()
        batch = (None, None, None, input_ids, None, input_mask, None, None)
        out = vbert(batch)
        self.assertEqual(out.shape, (2, 5))

    def test_VanillaUPA(self):
        vupa = VanillaUPA(n_user=100, n_product=200, n_classes=5, hidden_size=768)
        user_ids = (torch.rand(5)).long()
        product_ids = (torch.rand(5)).long()
        sentence_matrix = (torch.rand(10 * 5, 5)).long()
        batch = (user_ids, product_ids, None,
                 None, None, None, sentence_matrix, 5)
        out = vupa(batch)
        print(out.shape)
        #self.assertEqual(out.shape, (5, 5))


if __name__ == '__main__':
    unittest.main()
