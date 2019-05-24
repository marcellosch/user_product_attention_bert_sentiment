#! /usr/bin/python3

import unittest
from model import *

class TestModel(unittest.TestCase):
    def test_UserProductAttention_dimensions(self):
        upa = UserProductAttention()
        H = torch.rand(64,512,768)
        u = torch.rand(64,200)
        p = torch.rand(64,200)
        out = upa(H,u,p)
        self.assertEqual(out.shape,(64,768))

    def test_SimpleUserProductBert(self):
        supb = SimpleUserProductBert(n_user=100, n_product=200, n_classes=5)
        input_ids = (torch.rand(2,512)*800).long()
        input_mask = torch.ones(2,512).long()
        user_ids = (torch.rand(2) * 100).long()
        product_ids = (torch.rand(2)*200).long()
        out = supb(input_ids, input_mask, user_ids, product_ids)
        self.assertEqual(out.shape, (2,5))
        

if __name__ == '__main__':
    unittest.main()
