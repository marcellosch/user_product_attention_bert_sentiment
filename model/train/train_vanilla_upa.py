#! /usr/bin/python3
from model.train.train import parse_args, train
from utils.data import SentenceMatrixDataset
from model.vanilla_upa import VanillaUPA


if __name__ == "__main__":
    args = parse_args()

    folder = './data/' + args.dataset 

    userlist_filename = folder + '/usrlist.txt'
    productlist_filename = folder + '/prdlist.txt'
    wordlist_filename = folder + '/wordlist.txt'
    train_file = folder + '/train.txt'
    dev_file = folder + '/dev.txt'
    test_file = folder + '/test.txt'

    # Read training and test datasets
    train_dat = SentenceMatrixDataset(train_file, userlist_filename, productlist_filename,
                                 wordlist_filename, force_no_cache=args.force_document_processing)
    dev_dat = SentenceMatrixDataset(dev_file, userlist_filename, productlist_filename,
                               wordlist_filename, force_no_cache=args.force_document_processing)

    # Determine model parameter
    n_user = len(train_dat.users)
    n_product = len(train_dat.products)
    

    model = VanillaUPA(n_user, n_product, args.n_classes, args.user_size,
                       args.product_size, args.attention_hidden_size)
    train(model, train_dat, dev_dat, args, use_cat_collate=True)
