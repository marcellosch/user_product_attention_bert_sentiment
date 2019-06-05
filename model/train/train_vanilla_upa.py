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

    create_dataset = lambda doc_file: SentenceMatrixDataset(doc_file, userlist_filename, productlist_filename,
                        wordlist_filename, force_no_cache=args.force_document_processing)

    train_dat, dev_dat, test_dat = create_dataset(train_file), create_dataset(dev_file), create_dataset(test_file)

    n_user = len(train_dat.users)
    n_product = len(train_dat.products)
    n_classes = train_dat.get_n_classes()

    model = VanillaUPA(n_user, n_product, args.n_classes, args.user_size,
                       args.product_size, args.attention_hidden_size)
    train(model, train_dat, dev_dat, test_dat, args, use_cat_collate=True)
