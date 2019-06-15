''' Code to load trained models and evaluate on test datasets '''
from model.train.train import eval_on_data, save_args_to_file, save_results_to_file
import logging
import torch
from model.vanilla_bert import VanillaBert
from utils.data import SentimentDataset


def main(state_dict_path, ModelClass, data, out_path, model_args=(), model_kwargs={}, batch_size=1):

    print(model_args),
    print(model_kwargs)
    model = ModelClass(*model_args, **model_kwargs)
    logging.info("Reinititated the model.")

    model.load_state_dict(torch.load(state_dict_path))
    logging.info("Loaded the state dict.")

    no_cuda = False
    device = torch.device("cuda" if torch.cuda.is_available()
                          and not no_cuda else "cpu")
    model = model.to(device)

    acc, loss = eval_on_data(model, data, batch_size, device)
    logging.info("Evaluation results: Acc: {0}, Loss: {1}".format(acc, loss))
    save_results_to_file(out_path, test_results=[(acc, loss)])
    logging.info("Evaluation results saved to: {0}".format(out_path))


if __name__ == '__main__':

    dataset_name = "yelp13"

    folder = './data/' + dataset_name
    userlist_filename = folder + '/usrlist.txt'
    productlist_filename = folder + '/prdlist.txt'
    wordlist_filename = folder + '/wordlist.txt'
    test_file = folder + '/test.txt'

    test_dat = SentimentDataset(test_file, userlist_filename, productlist_filename,
                                wordlist_filename)

    # Example of run
    out_path = './'
    state_dict_path = './training_output/yelp13/vanilla_bet_0001/VanillaBert/pytorch_model.bin'
    main(state_dict_path, VanillaBert, test_dat, out_path, (5,))
