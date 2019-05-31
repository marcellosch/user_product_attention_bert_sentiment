import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from utils.data import SentimentDataset, userlist_filename, productlist_filename, wordlist_filename, train_file, test_file
from argparse import ArgumentParser
import logging
import random
import numpy as np
import pdb
from torch._utils.data._utils import default_collate

def cat_collate(batch):
    """ Concats the batches instead of stacking them like in the default_collate. """

    ret = torch.stack(l)
    pdb.set_trace()
    return ret

def eval_on_data(model, data, args, device, use_cat_collate=False):
    # Run prediction for full data
    sampler = SequentialSampler(data)

    if use_cat_collate:
        collate_fn = cat_collate
    else:
        collate_fn = default_collate
    
    eval_dataloader = DataLoader(data,
                                 sampler=sampler,
                                 batch_size=args.eval_batch_size,
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
        loss_function = CrossEntropyLoss()
        tmp_eval_loss = loss_function(logits, label.view(-1))

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)
        labels += label.tolist()

    preds = preds[0]
    preds = np.argmax(preds, axis=1)
    labels = np.array(labels)

    assert len(preds) == len(labels)
    accuracy = (preds == labels).mean()
    eval_loss = eval_loss / nb_eval_steps

    return accuracy, eval_loss


def parse_args():
    # Argument parsing
    parser = ArgumentParser()
    # parser.add_argument('--preprocessed_data', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of epochs to train for")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether or not to use CUDA")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--eval_batch_size", default=16, type=int)
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

    args = parser.parse_args()

    return args


def train(model, train_dat, dev_dat, args, use_cat_collate=False):
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
                        lr_this_step = args.learning_rate * warmup_linear.get_lr(global_step, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                # dev_acc, dev_loss = eval_on_data(model, dev_dat, args, device, n_classes)
                model.train()

        logging.info("***** Running evaluation on dev set *****")
        logging.info("  Num examples = %d", len(dev_dat))
        logging.info("  Batch size = %d", args.eval_batch_size)
        dev_acc, dev_loss = eval_on_data(model, dev_dat, args, device)
        logging.info(" Epoch = {0}, Accuracy = {1:.3f}, Loss = {2:.3f}".format(
            epoch, dev_acc, dev_loss))

    # Save a trained model
    logging.info("** ** * Saving fine-tuned model ** ** * ")
    model_to_save = model.module if hasattr(
        model, 'module') else model  # Only save the model it-self
    output_model_file = args.output_dir / "pytorch_model.bin"
    torch.save(model_to_save.state_dict(), str(output_model_file))
