""" Download the data from: http://www.thunlp.org/~chm/data/data.zip """

import torch
from torch.utils.data import Dataset
from collections import namedtuple
from pytorch_pretrained_bert.tokenization import BertTokenizer
import pickle
import os
import numpy as np
import zipfile
import wget


Word = namedtuple('Word', ['idx', 'id'])
Doc = namedtuple('Doc', ['user_id', 'product_id', 'label', 'text', 'sentence_idx', 'mask'])

CACHE_PATH = './cache/'
DATASET_URL = 'http://www.thunlp.org/~chm/data/data.zip'
    

class  SentimentDataset(Dataset):
    """ 
    Represents the sentiment dataset containing IMDB, Yelp13 and Yelp14. 
    Has to initialized for each set. Example:

    train_set = SentimentDataset(train_file, userlist_filename, productlist_filename, wordlist_filename)
    dev_set = SentimentDataset(dev_file, userlist_filename, productlist_filename, wordlist_filename)
    test_set = SentimentDataset(test_file, userlist_filename, productlist_filename, wordlist_filename)

    """

    def __init__(self, train_file, userlist_filename, productlist_filename, wordlist_filename, force_no_cache=False):
        
        if not os.path.exists('./data/yelp14'):
            wget.download(DATASET_URL)
            zf = zipfile.ZipFile(open("data.zip", "rb"))
            zf.extractall()

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', max_len=512)

        if not os.path.exists(CACHE_PATH):
            os.makedirs(CACHE_PATH)

        document_path = CACHE_PATH + 'documents'
        is_cached = os.path.isfile(document_path)
        if not is_cached or force_no_cache:
            self.users, self.user_string2int = self.read_userlist(userlist_filename)
            self.products, self.product_string2int = self.read_productlist(productlist_filename)
            self.word_list, self.vocabulary = self.read_vocabulary(wordlist_filename)
            self.document_list, self.documents = self.read_documents(train_file, document_path)
            print("Preprocessed {0} documents and cached to disk.".format(len(self.documents)))
        else:
            self.documents = self.read_docs_from_cache(document_path)
            print("Loaded {0} documents from disk.".format(len(self.documents)))


    def preprocess(self, text, sentence_delimeter='.'):
        """
        Takes text as input and does the:
            - tokenization
            - token to id conversion
            - calculate sentence positions
            - calculate the mask
        """
        text = text.replace(sentence_delimeter, '[SEP]')
        tokenized = self.tokenizer.tokenize(text)
        tokenized = (['[CLS]'] + tokenized + 512 * ['[PAD]'])[:512]

        sentence_idx = []
        begin, end = 0,0
        for token in tokenized:
            end += 1
            if token == '[SEP]':
                sentence_idx.append((begin, end))
                begin, end = end, end
        sentence_idx.append((begin, end))            
        tokenized = [token for token in tokenized if token != '[SEP]']
        token_ids = self.tokenizer.convert_tokens_to_ids(tokenized)
        mask = [1 if token != 0 else 0 for token in token_ids]
        return token_ids, sentence_idx, mask


    def read_userlist(self, filename):
        """ Read userlist from file containing one user id per line. """
        lines = list(map(lambda x: x.split(),
                open(userlist_filename).readlines()))
        user_list = [item[0] for item in lines]
        
        unique_users = list(set(user_list))
        string2int_id = {string_id: int_id for int_id, string_id in enumerate(unique_users)}

        return user_list, string2int_id


    def read_productlist(self, filename):
        """ Read product list from file containing one product id per line. """
        lines = list(map(lambda x: x.split(),
                open(filename).readlines()))
        product_list = [item[0] for item in lines]

        unique_products = list(set(product_list))
        string2int_id = {string_id: int_id for int_id, string_id in enumerate(unique_products)}

        return product_list, string2int_id


    def read_vocabulary(self, filename):
        """ Read vocabulary from file containing one word per line. """
        lines = list(map(lambda x: x.split(),
                open(wordlist_filename).readlines()))

        word_list = [item[0] for item in lines]
        vocab = {word: idx for idx, word in enumerate(word_list)}

        return word_list, vocab


    def read_documents(self, filename, cache_path):
        """
        Read reviews from file with each line containing:
        - user_id 
        - product_id 
        - review text seperated with double tabs('\t\t').
        """

        # limit the amount of documents for testing purposes if necessary
        # lines = list(map(lambda x: x.split('\t\t'), open(train_file).readlines()))[:100] 
        lines = list(map(lambda x: x.split('\t\t'), open(train_file).readlines()))
        documents = []
        document_list = []
        self.count_long_text = 0
        for i, line in enumerate(lines):
            user_id, product_id, label, text = line

            text, sentence_idx, mask = self.preprocess(text, sentence_delimeter='<sssss>')
            document_list.append((user_id, product_id, label, text))
            
            user_id = torch.tensor(self.user_string2int[user_id], dtype=torch.int64)
            product_id = torch.tensor(self.product_string2int[product_id], dtype=torch.int64)
            label = torch.tensor(int(label), dtype=torch.int64)
            text = torch.tensor(text, dtype=torch.int64)
            sentence_idx = torch.tensor(sentence_idx, dtype=torch.int64)
            mask = torch.tensor(mask, dtype=torch.int64)
            doc = Doc(user_id=user_id,
                      product_id=product_id,
                      label=label,
                      text=text, 
                      sentence_idx=sentence_idx,
                      mask=mask)

            documents.append(doc)

            if i % 5000 == 0:
                print("Processed {0} of {1} documents. ({2:.1f}%)".format(i, len(lines), i*100/len(lines)))

        pickle.dump(documents, open(cache_path, "wb"))
        return document_list, documents


    def read_docs_from_cache(self, load_path):
        """ Loads the cached preprocessed documents from disk. """
        return pickle.load(open(load_path, "rb"))


    def __getitem__(self, idx):
       doc = self.documents[idx]
       return doc

    def __len__(self):
        return len(self.documents)
    

userlist_filename = './data/yelp14/usrlist.txt'
productlist_filename = './data/yelp14/prdlist.txt'
wordlist_filename = './data/yelp14/wordlist.txt'
train_file = './data/yelp14/train.txt'
dev_file = './data/yelp14/dev.txt'
test_file = './data/yelp14/test.txt'


if __name__ == '__main__':
    """ Just to test. """
    ds = SentimentDataset(train_file, userlist_filename, productlist_filename, wordlist_filename)
    print(ds[0])