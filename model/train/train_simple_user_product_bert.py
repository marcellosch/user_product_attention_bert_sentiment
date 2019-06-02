#! /usr/bin/python3
from model.train.train import parse_args, SentimentDataset, train
from model.simple_upa_bert import SimpleUPABert


if __name__ == "__main__":
    args = parse_args()

    folder = './data/' + args.dataset 

    userlist_filename = folder + '/usrlist.txt'
    productlist_filename = folder + '/prdlist.txt'
    wordlist_filename = folder + '/wordlist.txt'
    train_file = folder + '/train.txt'
    dev_file = folder + '/dev.txt'
    test_file = folder + '/test.txt'

    train_dat = SentimentDataset(train_file, userlist_filename, productlist_filename,
                                 wordlist_filename, force_no_cache=args.force_document_processing)
    dev_dat = SentimentDataset(test_file, userlist_filename, productlist_filename,
                               wordlist_filename, force_no_cache=args.force_document_processing)

    n_user = len(train_dat.users)
    n_product = len(train_dat.products)
    n_classes = 5

    model = SimpleUPABert(n_user, n_product, n_classes, args.user_size,
                          args.product_size, args.attention_hidden_size)
    train(model, train_dat, dev_dat, args)
