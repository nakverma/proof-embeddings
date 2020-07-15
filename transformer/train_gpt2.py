import os

import torch
from torch.utils.data import DataLoader

from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel

from utils.data_utils import LogicDataset, LogicMLMDataCollator
from utils.model_utils import train, test, get_optimizer_and_scheduler


# Configure device
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('DEVICE: ', DEVICE)

# Set hyperparameters
BATCH_SIZE = 32
SPLIT_RATIO = 0.33
SEED = 42

# NOTE: Sequence length is now based on 6 expressions, as opposed to the previous 2 expressions
#       in the train_bert and test_bert scripts
SEQUENCE_LENGTH = 150
HIDDEN_SIZE = 192
NUM_HIDDEN_LAYERS = 6
NUM_ATTENTION_HEADS = 6

BETAS = (0.9, 0.999)
WEIGHT_DECAY = 0.01
EPS = 1e-8

# NOTE: We give a low mask probability since we don't want to bias the language model in
#       decoding to (i.e. generating) [MASK] tokens -- in fact, we have no use of them
MASK_PROBABILITY = 0.075
NUM_EPOCHS_LM = 50
LEARNING_RATE_LM = 5E-5


# Configure and create save paths for generative model
GENERATIVE_SAVE_DIR = os.path.join('saved_models', 'generative')
if not os.path.exists(GENERATIVE_SAVE_DIR):
    os.makedirs(GENERATIVE_SAVE_DIR)

# Configure paths to .txt files for vocabulary (and merges) and get vocabulary size
# NOTE: For the BERT model, we were using .txt but for GPT2 we will use .json
VOCABULARY_PATH = os.path.join('data', 'vocabulary.json')
VOCABULARY_SIZE = len(open(VOCABULARY_PATH, 'r').readlines())
MERGES_PATH = os.path.join('data', 'merges.txt')
# Configure path to .pkl file for actual logic tree data
DATA_PATH = os.path.join('data', 'T_flat_unigram_dataset.pkl')

# Configure the tokenizer
TOKENIZER = GPT2Tokenizer(vocab_file=VOCABULARY_PATH,
                          merges_file=MERGES_PATH,
                          errors='replace',
                          unk_token='[UNK]',
                          bos_token='[BOS]',
                          eos_token='[EOS]',
                          pad_token='[PAD]',
                          sep_token='[SEP]',
                          mask_token='[MASK]')

# Get (supervised) datasets and data collator for masked-language-modeling
train_dataset = LogicDataset(data_path=DATA_PATH,
                             tokenizer=TOKENIZER,
                             sequence_length=SEQUENCE_LENGTH,
                             split='train',
                             split_ratio=SPLIT_RATIO,
                             supervision_mode='self_supervised',
                             data_mode='sequential',
                             tokenization_mode='gpt2',
                             seed=SEED)
test_dataset = LogicDataset(data_path=DATA_PATH,
                            tokenizer=TOKENIZER,
                            sequence_length=SEQUENCE_LENGTH,
                            split='test',
                            split_ratio=SPLIT_RATIO,
                            supervision_mode='self_supervised',
                            data_mode='sequential',
                            tokenization_mode='gpt2',
                            seed=SEED)

data_collator = LogicMLMDataCollator(tokenizer=TOKENIZER, mask_probability=MASK_PROBABILITY)
print('Num Training Examples: %d \t Num. Testing Examples: %d' % (len(train_dataset), len(test_dataset)))

# Get dataloaders
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          collate_fn=data_collator.collate_batch,
                          pin_memory=True,
                          shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         collate_fn=data_collator.collate_batch,
                         pin_memory=True,
                         shuffle=False)

# Configure our own custom GPT2
config = GPT2Config(vocab_size=VOCABULARY_SIZE,
                    n_positions=SEQUENCE_LENGTH,
                    n_ctx=HIDDEN_SIZE,
                    n_embd=HIDDEN_SIZE,
                    n_layer=NUM_HIDDEN_LAYERS,
                    n_head=NUM_ATTENTION_HEADS,
                    bos_token_id=TOKENIZER.bos_token_id,
                    eos_token_id=TOKENIZER.eos_token_id,
                    mask_token_id=TOKENIZER.mask_token_id,
                    pad_token_id=TOKENIZER.pad_token_id)
model = GPT2LMHeadModel(config=config)
print('GPT2 Num. Parameters: %d' % model.num_parameters())
# Place model on device
model = model.to(DEVICE)
# Initialize optimizer and scheduler for MLM task
optimizer, scheduler = get_optimizer_and_scheduler(model=model,
                                                   learning_rate=LEARNING_RATE_LM,
                                                   betas=BETAS,
                                                   weight_decay=WEIGHT_DECAY,
                                                   eps=EPS,
                                                   num_warmup_steps=50,
                                                   num_training_steps=int(NUM_EPOCHS_LM*len(train_dataset)/BATCH_SIZE))

print('--------------PRETRAINING----------------')
# (1) PRETRAINING ON LANGUAGE MODELING -- this is all that we'll do with GPT2 anyway
# NOTE: This is not masked language modeling (MLM), but conventional language modeling (LM).
#       This doesn't mean we don't mask; we still mask to perturb the data as a data-aug. strategy.
#       Essentially the input is masked, but the labels are not the actual values this time,
#       instead it is a one-to-one copy of the input, which is shifted L-to-R inside the model.
# Start actual training, check test loss after each epoch
best_test_loss = float('inf')
for epoch in range(NUM_EPOCHS_LM):
    print("EPOCH NO: %d" % (epoch + 1))

    train_loss, _ = train(model=model,
                          scheduler=scheduler,
                          tokenizer=TOKENIZER,
                          iterator=train_loader,
                          optimizer=optimizer,
                          mode='language_modeling',
                          device=DEVICE)
    test_loss, _ = test(model=model,
                        tokenizer=TOKENIZER,
                        iterator=test_loader,
                        mode='language_modeling',
                        device=DEVICE)

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        model.save_pretrained(GENERATIVE_SAVE_DIR)

    print(f'\tTrain Loss: {train_loss:.3f}')
    print(f'\tTest Loss:  {test_loss:.3f}')
