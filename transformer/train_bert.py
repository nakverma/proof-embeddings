import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import BertTokenizer, BertConfig, BertForMaskedLM, BertForSequenceClassification

from utils.data_utils import LogicDataset, LogicMLMDataCollator
from utils.model_utils import train, test, get_optimizer_and_scheduler


# Configure device
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('DEVICE: ', DEVICE)

# Set hyperparameters
BATCH_SIZE = 32
SPLIT_RATIO = 0.33
SEED = 42

PRETRAIN = False  # True: performs pretraining (MLM) | False: performs finetuning (classification)

# TODO: This sequence length might be need to be updated -- we are not counting [SEP], [CLS], etc.
SEQUENCE_LENGTH = 48
HIDDEN_SIZE = 192
NUM_HIDDEN_LAYERS = 6
NUM_ATTENTION_HEADS = 6

BETAS = (0.9, 0.999)
WEIGHT_DECAY = 0.01
EPS = 1e-8

MASK_PROBABILITY = 0.25
NUM_EPOCHS_MLM = 50
LEARNING_RATE_MLM = 5E-5

NUM_EPOCHS_CLS = 50
LEARNING_RATE_CLS = 5E-5

# Configure and create save paths for i) pretrained MLM and ii) finetuned classification models
PRETRAINED_SAVE_DIR = os.path.join('saved_models', 'pretrained')
FINETUNED_SAVE_DIR = os.path.join('saved_models', 'finetuned')
if not os.path.exists(PRETRAINED_SAVE_DIR):
    os.makedirs(PRETRAINED_SAVE_DIR)
if not os.path.exists(FINETUNED_SAVE_DIR):
    os.makedirs(FINETUNED_SAVE_DIR)

# Configure paths to .txt files for vocabulary and get vocabulary size
VOCABULARY_PATH = os.path.join('data', 'vocabulary.txt')
VOCABULARY_SIZE = len(open(VOCABULARY_PATH, 'r').readlines())
# Configure path to .pkl file for actual logic tree data
DATA_PATH = os.path.join('data', 'T_flat_unigram_dataset.pkl')

# Configure the tokenizer
TOKENIZER = BertTokenizer(vocab_file=VOCABULARY_PATH,
                          model_max_length=SEQUENCE_LENGTH,
                          do_lower_case=False,  # NOTE: Don't lower-case for T and F
                          do_basic_tokenize=True,
                          unk_token='[UNK]',
                          sep_token='[SEP]',
                          pad_token='[PAD]',
                          cls_token='[CLS]',
                          mask_token='[MASK]')

if PRETRAIN:
    # Get (supervised) datasets and data collator for masked-language-modeling
    train_dataset = LogicDataset(data_path=DATA_PATH,
                                 tokenizer=TOKENIZER,
                                 sequence_length=SEQUENCE_LENGTH,
                                 split='train',
                                 split_ratio=SPLIT_RATIO,
                                 supervision_mode='self_supervised',
                                 data_mode='pairs',
                                 seed=SEED)
    test_dataset = LogicDataset(data_path=DATA_PATH,
                                tokenizer=TOKENIZER,
                                sequence_length=SEQUENCE_LENGTH,
                                split='test',
                                split_ratio=SPLIT_RATIO,
                                supervision_mode='self_supervised',
                                data_mode='pairs',
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

    # Configure our own custom BERT
    config = BertConfig(vocab_size=VOCABULARY_SIZE,
                        hidden_size=HIDDEN_SIZE,
                        num_hidden_layers=NUM_HIDDEN_LAYERS,
                        num_attention_heads=NUM_ATTENTION_HEADS,
                        max_position_embeddings=SEQUENCE_LENGTH)
    model = BertForMaskedLM(config=config)
    print('BERT Num. Parameters: %d' % model.num_parameters())
    # Place model on device
    model = model.to(DEVICE)

    # Initialize optimizer and scheduler for MLM task
    optimizer, scheduler = get_optimizer_and_scheduler(model=model,
                                                       learning_rate=LEARNING_RATE_MLM,
                                                       betas=BETAS,
                                                       weight_decay=WEIGHT_DECAY,
                                                       eps=EPS,
                                                       num_warmup_steps=50,
                                                       num_training_steps=int(NUM_EPOCHS_MLM*len(train_dataset)/BATCH_SIZE))

    print('--------------PRETRAINING----------------')
    # (1) PRETRAINING ON MASKED LANGUAGE MODELING
    # Start actual training, check test loss after each epoch
    best_test_loss = float('inf')
    for epoch in range(NUM_EPOCHS_MLM):
        print("EPOCH NO: %d" % (epoch + 1))

        train_loss, _ = train(model=model,
                              scheduler=scheduler,
                              tokenizer=TOKENIZER,
                              iterator=train_loader,
                              optimizer=optimizer,
                              mode='masked_language_modeling',
                              device=DEVICE)
        test_loss, _ = test(model=model,
                            tokenizer=TOKENIZER,
                            iterator=test_loader,
                            mode='masked_language_modeling',
                            device=DEVICE)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            model.save_pretrained(PRETRAINED_SAVE_DIR)

        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\tTest Loss:  {test_loss:.3f}')
else:
    assert os.path.exists(os.path.join(PRETRAINED_SAVE_DIR, 'pytorch_model.bin'))
    print('Skipping Pretraining -- will use previously saved model!')

print('--------------FINETUNING----------------')
# (2) FINETUNING ON SEQUENCE CLASSIFICATION
# Define loss function for classification
criterion = nn.CrossEntropyLoss()
# Get (supervised) datasets
train_dataset = LogicDataset(data_path=DATA_PATH,
                             tokenizer=TOKENIZER,
                             sequence_length=SEQUENCE_LENGTH,
                             split='train',
                             split_ratio=SPLIT_RATIO,
                             supervision_mode='supervised',
                             data_mode='pairs',
                             seed=SEED)
test_dataset = LogicDataset(data_path=DATA_PATH,
                            tokenizer=TOKENIZER,
                            sequence_length=SEQUENCE_LENGTH,
                            split='test',
                            split_ratio=SPLIT_RATIO,
                            supervision_mode='supervised',
                            data_mode='pairs',
                            seed=SEED)
print('Num Training Examples: %d \t Num. Testing Examples: %d' % (len(train_dataset), len(test_dataset)))

# Get dataloaders
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          pin_memory=True,
                          shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         pin_memory=True,
                         shuffle=False)

# Load pretrained model with correct config - we have to add num. labels for classification task
assert train_dataset.get_num_labels() == test_dataset.get_num_labels()
model = BertForSequenceClassification.from_pretrained(PRETRAINED_SAVE_DIR,
                                                      num_labels=train_dataset.get_num_labels())
print('BERT Num. Parameters: %d' % model.num_parameters())
# Place model on device
model = model.to(DEVICE)

# Initialize optimizer and scheduler for CLS task
optimizer, scheduler = get_optimizer_and_scheduler(model=model,
                                                   learning_rate=LEARNING_RATE_CLS,
                                                   betas=BETAS,
                                                   weight_decay=WEIGHT_DECAY,
                                                   eps=EPS,
                                                   num_warmup_steps=50,
                                                   num_training_steps=int(NUM_EPOCHS_CLS*len(train_dataset)/BATCH_SIZE))

best_test_loss = float('inf')
for epoch in range(NUM_EPOCHS_CLS):
    print("EPOCH NO: %d" % (epoch + 1))

    train_loss, train_accuracy = train(model=model,
                                       scheduler=scheduler,
                                       tokenizer=TOKENIZER,
                                       iterator=train_loader,
                                       optimizer=optimizer,
                                       mode='sequence_classification',
                                       device=DEVICE)
    test_loss, test_accuracy = test(model=model,
                                    tokenizer=TOKENIZER,
                                    iterator=test_loader,
                                    mode='sequence_classification',
                                    device=DEVICE)

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save(model.state_dict(), os.path.join(FINETUNED_SAVE_DIR, 'step_classifier.pt'))

    print(f'\tTrain Loss: {train_loss:.3f} | Train Accuracy: {train_accuracy * 100:.2f}%')
    print(f'\tTest Loss:  {test_loss:.3f} | Test Accuracy:  {test_accuracy * 100:.2f}%')
