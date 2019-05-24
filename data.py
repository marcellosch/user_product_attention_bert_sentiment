"""
Download the data from: http://www.thunlp.org/~chm/data/data.zip
"""


from torch.utils.data import Dataset
from collections import namedtuple
from pytorch_pretrained_bert.tokenization import BertTokenizer

Word = namedtuple('Word', ['idx', 'id'])
Doc = namedtuple('Doc', ['user_id', 'product_id', 'label', 'text', 'sentence_idx', 'mask'])

class  SentimentDataset(Dataset):
    

    def __init__(self, train_file, userlist_filename, productlist_filename, wordlist_filename):
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', max_len=512)
   
        self.users, self.user_string2int = self.read_userlist(userlist_filename)
        self.products, self.product_string2int = self.read_productlist(productlist_filename)
        self.word_list, self.vocabulary = self.read_vocabulary(wordlist_filename)
        self.document_list, self.documents = self.read_documents(train_file)
    

    """
    Takes text as input and does the:
        - tokenization
        - token to id conversion
        - calculate sentence positions
        - calculate the mask
    """
    def preprocess(self, text, sentence_delimeter='.'):
        text = text.replace(sentence_delimeter, '[SEP]')
        tokenized = self.tokenizer.tokenize(text)
        tokenized = (tokenized + 512 * ['[PAD]'])[:512]

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


    """
    Read userlist from file containing one user id per line.
    """
    def read_userlist(self, filename):
        lines = list(map(lambda x: x.split(),
                open(userlist_filename).readlines()))
        user_list = [item[0] for item in lines]
        
        unique_users = list(set(user_list))
        string2int_id = {string_id: int_id for int_id, string_id in enumerate(unique_users)}

        return user_list, string2int_id


    """
    Read product list from file containing one product id per line.
    """
    def read_productlist(self, filename):
        lines = list(map(lambda x: x.split(),
                open(filename).readlines()))
        product_list = [item[0] for item in lines]

        unique_products = list(set(product_list))
        string2int_id = {string_id: int_id for int_id, string_id in enumerate(unique_products)}

        return product_list, string2int_id


    """
    Read vocabulary from file containing one word per line.
    """
    def read_vocabulary(self, filename):
        lines = list(map(lambda x: x.split(),
                open(wordlist_filename).readlines()))

        word_list = [item[0] for item in lines]
        vocab = {word: idx for idx, word in enumerate(word_list)}

        return word_list, vocab


    """
    Read reviews from file with each line containing:
     - user_id 
     - product_id 
     - review text seperated with double tabs('\t\t').
    """
    def read_documents(self, filename):

        lines = list(map(lambda x: x.split('\t\t'), open(train_file).readlines()))[:1000] # [:100] just for testing purpose
        documents = []
        document_list = []
        self.count_long_text = 0
        for line in lines:
            user_id, product_id, label, text = line

            text, sentence_idx, mask = self.preprocess(text, sentence_delimeter='<sssss>')
            document_list.append((user_id, product_id, label, text))
            documents.append(Doc(user_id=self.user_string2int[user_id],
                                 product_id=self.product_string2int[product_id],
                                 label=int(label),
                                 text=text, 
                                 sentence_idx=sentence_idx,
                                 mask=mask))

        return document_list, documents


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

ds = SentimentDataset(train_file, userlist_filename, productlist_filename, wordlist_filename)
print(ds[0].user_id)