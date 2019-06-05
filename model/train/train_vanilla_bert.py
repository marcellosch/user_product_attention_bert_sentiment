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

    create_dataset = lambda doc_file: SentimentDataset(doc_file, userlist_filename, productlist_filename,
                            wordlist_filename, force_no_cache=args.force_document_processing)

    train_dat, dev_dat, test_dat = create_dataset(train_file), create_dataset(dev_file), create_dataset(test_file)

    # Determine model parameter
    n_classes = train_dat.get_n_classes()

    model = VanillaBert(n_classes)
    train(model, train_dat, dev_dat, test_dat, args)
