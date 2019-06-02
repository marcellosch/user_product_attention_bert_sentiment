#! /usr/bin/python3
from model.train.train import parse_args, SentimentDataset, train
from model.vanilla_bert import VanillaBert


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

    # Determine model parameter
    n_classes = train_dat.get_n_classes()

    model = VanillaBert(n_classes)
    train(model, train_dat, dev_dat, args)
