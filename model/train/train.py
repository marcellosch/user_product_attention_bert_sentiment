import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, Subset
from tqdm import tqdm
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from utils.data import SentimentDataset
from argparse import ArgumentParser
import logging
import random
import numpy as np
from torch.utils.data._utils.collate import default_collate
import json
import os
from pathlib import Path


def cat_collate(batch):
    """ Concats the batches instead of stacking them like in the default_collate. """
    max_words_in_sentence = max([b[3].shape[1] for b in batch])
    max_sentences_in_docs = max([b[3].shape[0] for b in batch])
    sentence_matrix = torch.zeros(
        len(batch) * max_sentences_in_docs, max_words_in_sentence, dtype=torch.int64)
    for i, doc in enumerate(batch):
        height, width = doc[3].shape
        begin_row = i*max_sentences_in_docs
        end_row = begin_row + height
        end_col = width
        sentence_matrix[begin_row:end_row, 0:end_col] = doc[3]

    user_id = torch.tensor([b[0] for b in batch], dtype=torch.int64)
    product_id = torch.tensor([b[1] for b in batch], dtype=torch.int64)
    label = torch.tensor([b[2] for b in batch], dtype=torch.int64)
    return (user_id, product_id, label, sentence_matrix)


def eval_on_data(model, data, batch_size, device, use_cat_collate=False, step=None):
    # Run prediction for full data

    if use_cat_collate:
        collate_fn = cat_collate
    else:
        collate_fn = default_collate

    if not step is None:
        inds = np.random.choice(range(len(data)), 2000)
        data = Subset(data, inds)

    eval_dataloader = DataLoader(data,
                                 batch_size=batch_size,
                                 collate_fn=collate_fn)

    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    labels = []

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)
        label = batch[2]
        #user_id, product_id, label, text, sentence_idx, mask = batch

        #user_id, product_id, label, text, sentence_idx, mask = batch
        #user_id, product_id, label, text, sentence_idx, mask = user_id.to(device), product_id.to(device), label.to(device), text.to(device), sentence_idx.to(device), mask.to(device)

        with torch.no_grad():
            logits = model(batch)

        # create eval loss
        loss_function = torch.nn.NLLLoss()
        tmp_eval_loss = loss_function(logits, label.view(-1))

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1

        # if len(preds) == 0:
        #     preds.append(logits.detach().cpu().numpy())
        # else:
        #     preds[0] = np.append(
        #         preds[0], logits.detach().cpu().numpy(), axis=0)
        preds += logits.cpu().argmax(dim=1).tolist()
        labels += label.tolist()

    #preds = preds[0]
    preds = np.array(preds)
    labels = np.array(labels)

    assert len(preds) == len(labels)
    accuracy = (preds == labels).mean()
    eval_loss = eval_loss / nb_eval_steps

    return accuracy, eval_loss


def parse_args():
    # Argument parsing
    parser = ArgumentParser()
    # parser.add_argument('--preprocessed_data', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path,
                        default=Path("training_output"))
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of epochs to train for")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether or not to use CUDA")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--eval_batch_size", default=32, type=int)
    parser.add_argument("--fp16", action='store_true')
    parser.add_argument("--loss_scale", type=float, default=0)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--learning_rate", default=3e-5, type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--user_size", type=int, default=200,
                        help="User embedding dimension")
    parser.add_argument("--product_size", type=int,
                        default=200, help="Product embedding dimension")
    parser.add_argument("--attention_hidden_size", type=int,
                        default=200, help="Attention hidden state dimension")
    parser.add_argument("--force_document_processing",
                        action="store_true", help="Force document preprocessing")
    parser.add_argument("--dataset", default="yelp14", type=str)
    parser.add_argument("--n_classes", default=5, type=int)
    args = parser.parse_args()

    return args


def train(model, train_dat, dev_dat, args, use_cat_collate=False):

    train_results = []
    dev_results = []
    test_results = []

    out_folder = args.output_dir / model.__class__.__name__ / \
        args.dataset / str(args.learning_rate).split(".")[-1]
    out_folder.mkdir(parents=True, exist_ok=True)
    out_args_path = out_folder / "args.json"
    out_results_path = out_folder / "results.json"
    save_args_to_file(out_args_path, args)

    log_format = '%(asctime)-10s: %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format)

    # Initialize model
    if args.fp16:
        model.half()

    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available()
                          and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logging.info("device: {} n_gpu: {}".format(device, n_gpu))
    # TODO: check out distributed training!

    # Adjust train_batch size if gradients should be accumulated
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Input passed random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Create output dir if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate total training sample number:
    total_train_examples = args.epochs * len(train_dat)
    num_train_optimization_steps = int(
        total_train_examples / args.train_batch_size / args.gradient_accumulation_steps)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    # TODO read about weight decay, do we want this to happen in our model?
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("apex not installed")
        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(
                optimizer, static_loss_scale=args.loss_scale)
            warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
                                                 t_total=num_train_optimization_steps)
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    criterion = torch.nn.NLLLoss()

    global_step = 0
    logging.info("***** Running training *****")
    logging.info(f"  Num examples = {total_train_examples}")
    logging.info("  Batch size = %d", args.train_batch_size)
    logging.info("  Num steps = %d", num_train_optimization_steps)
    model = model.to(device)
    model.train()

    if use_cat_collate:
        collate_fn = cat_collate
    else:
        collate_fn = default_collate

    for epoch in range(args.epochs):
        train_sampler = RandomSampler(train_dat)
        train_dataloader = DataLoader(train_dat,
                                      sampler=train_sampler,
                                      batch_size=args.train_batch_size,
                                      collate_fn=collate_fn)
        tr_loss = 0
        nb_tr_steps = 0
        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}") as pbar:
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
               # user_id, product_id, label, text, sentence_idx, mask = batch
                #user_id, product_id, label, text, sentence_idx, mask = user_id.to(device), product_id.to(device), label.to(device), text.to(device), sentence_idx.to(device), mask.to(device)

                prediction = model(batch)
                label = batch[2]
                #prediction = model(text, mask, user_id, product_id)

                loss = criterion(prediction, label)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                tr_loss += loss.item()
                nb_tr_steps += 1
                pbar.update(1)
                mean_loss = tr_loss * args.gradient_accumulation_steps / nb_tr_steps
                pbar.set_postfix_str(f"Loss: {mean_loss:.5f}")
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * \
                            warmup_linear.get_lr(
                                global_step, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                if (step+1) % 20000 == 0:
                    dev_acc, dev_loss = eval_on_data(
                        model, dev_dat, args.eval_batch_size, device, use_cat_collate=use_cat_collate, step=step)
                    train_acc, train_loss = eval_on_data(
                        model, train_dat, args.eval_batch_size, device, use_cat_collate=use_cat_collate, step=step)
                    logging.info("Step: {} Training loss: {}, acc: {}, Dev loss: {}, acc: {}\n\n".format(
                        step, train_loss, train_acc, dev_loss, dev_acc))
                    model.train()

        logging.info("***** Running evaluation on train set *****")
        logging.info("  Num examples = %d", len(train_dat))
        logging.info("  Batch size = %d", args.train_batch_size)

        train_acc, train_loss = eval_on_data(
            model, train_dat, args.train_batch_size, device, use_cat_collate=use_cat_collate, step=step)
        logging.info(" Epoch = {0}, Accuracy = {1:.3f}, Loss = {2:.3f}".format(
            epoch, train_acc, train_loss))
        train_results.append((train_acc, train_loss))

        logging.info("***** Running evaluation on dev set *****")
        logging.info("  Num examples = %d", len(dev_dat))
        logging.info("  Batch size = %d", args.eval_batch_size)

        dev_acc, dev_loss = eval_on_data(
            model, dev_dat, args.train_batch_size, device, use_cat_collate=use_cat_collate)
        logging.info(" Epoch = {0}, Accuracy = {1:.3f}, Loss = {2:.3f}".format(
            epoch, dev_acc, dev_loss))
        dev_results.append((dev_acc, dev_loss))

        save_results_to_file(out_results_path, train_results,
                             dev_results, test_results)

    # Save a trained model
    logging.info("** ** * Saving fine-tuned model ** ** * ")
    model_to_save = model.module if hasattr(
        model, 'module') else model  # Only save the model it-self
    output_model_file = out_folder / "pytorch_model.bin"
    torch.save(model_to_save.state_dict(), str(output_model_file))

    save_results_to_file(out_results_path, train_results,
                         dev_results, test_results)


def save_results_to_file(path, train_results=[], dev_results=[], test_results=[]):
    results = {
        'train_data': train_results,
        'dev_data': dev_results,
        'test_data': test_results
    }
    with open(path, 'w') as f:
        json.dump(results, f)

 
def save_args_to_file(path, args):
    args_dict = vars(args)
    with open(path, 'w') as f:
        json.dump({k: str(v) for k, v in args_dict.items()}, f)
