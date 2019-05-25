from pathlib import Path
from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm import tqdm
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from model_simple_user_product_bert import SimpleUserProductBert
from data import SentimentDataset, userlist_filename, productlist_filename, wordlist_filename, train_file


# Argument parsing
parser = ArgumentParser()
parser.add_argument('--preprocessed_data', type=Path, required=True)
parser.add_argument('--output_dir', type=Path, required=True)
parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train for")
parser.add_argument("--no_cuda", action='store_true', help="Wheter to use CUDA")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--train_batch_size", default=16, type=int)
parser.add_argument("--fp16", action='store_true')
parser.add_argument("--loss_scale", type=float, default=0)
parser.add_argument("--warmup_proportion", default=0.1, type=float)
parser.add_argument("--learning_rate", default=3e-5, type=float)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--user_size", type=int, default=200, help="User embedding dimension")
parser.add_argument("--product_size", type=int, default=200, help="Product embedding dimension")
parser.add_argument("--attention_hidden_size", type=int, default=200, help="Attention hidden state dimension")

args = parser.parse_args()

# Read training and test datasets
train_dat = SentimentDataset(train_file, userlist_filename, productlist_filename, wordlist_filename)
test_dat = SentimentDataset(test_file, userlist_filename, productlist_filename, wordlist_filename)

# Determine model parameter
n_user = len(train_dat.users)
n_product = len(train_dat.products)
n_classes = 5

# Initialize model
model = SimpleUserProductBert(n_user, n_product, n_classes, args.user_size, args.product_size, args.attention_hidden_size)
if args.fp16:
    model.half()

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
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
total_train_examples = len(train_dat) 
num_train_optimization_steps = int(total_train_examples / args.batch_size / args.gradient_accumulation_steps)

# Prepare optimizer
param_optimizer = list(model.named_parameters())
# TODO read about weight decay, do we want this to happen in our model?
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
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
        optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
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
model.train()

for epoch in range(args.epochs):
    train_sampler = RandomSampler(train_dat)
    train_dataloader = DataLoader(train_dat, sampler=train_sampler, batch_size=args.train_batch_size)
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}") as pbar:
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, lm_label_ids, target_class = batch
            tr_loss = criterion(model(input_ids, segment_ids, input_mask, lm_label_ids), target_class)
            if n_gpu > 1:
                tr_loss = loss.mean() # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                tr_loss = loss / args.gradient_accumulation_steps
            else:
                tr_loss.backward()
            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            pbar.update(1)
            mean_loss = tr_loss * args.gradient_accumulation_steps / nb_tr_steps
            pbar.set_postfix_str(f"Loss: {mean_loss:.5f}")
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            if test_dat and (step + 1) % 1000 == 0:
                test_loss = 0
                test_sampler = RandomSampler(test_dat)
                test_dataloader = DataLoader(test_dat, sampler=test_sampler, batch_size=args.train_batch_size)
                for step, batch in enumerate(test_dataloader)
                    pass
                    #batch = tuple(t.to(device) for t in batch)
                    # TODO: write code that evaluates model performance on test set
                

# Save a trained model
logging.info("** ** * Saving fine-tuned model ** ** * ")
model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
output_model_file = args.output_dir / "pytorch_model.bin"
torch.save(model_to_save.state_dict(), str(output_model_file))


