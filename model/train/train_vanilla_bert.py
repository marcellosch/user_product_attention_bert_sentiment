#! /usr/bin/python3
from model.train.train import parse_args, SentimentDataset, train
from model.vanilla_bert import VanillaBert


if __name__ == "__main__":
    args = parse_args()

    userlist_filename = './data/yelp14/usrlist.txt'
    productlist_filename = './data/yelp14/prdlist.txt'
    wordlist_filename = './data/yelp14/wordlist.txt'
    train_file = './data/yelp14/train.txt'
    dev_file = './data/yelp14/dev.txt'
    test_file = './data/yelp14/test.txt'

    # Read training and test datasets
    train_dat = SentimentDataset(train_file, userlist_filename, productlist_filename,
                                 wordlist_filename, force_no_cache=args.force_document_processing)
    dev_dat = SentimentDataset(test_file, userlist_filename, productlist_filename,
                               wordlist_filename, force_no_cache=args.force_document_processing)

    # Determine model parameter
    n_classes = 5

    model = VanillaBert(n_classes)
    train(model, train_dat, dev_dat)
