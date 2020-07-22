import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import BertTokenizer, BertConfig, BertForMaskedLM, BertForSequenceClassification

from utils.data_utils import LogicDataset, LogicMLMDataCollator
from utils.model_utils import train, test, get_optimizer_and_scheduler

# Variables for data ops.
parser = argparse.ArgumentParser(description='Script to train a BERT model on logic data')
parser.add_argument('--seed', type=int, default=42, help='Set seeds for reproducibility')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size to be used in training and testing')
parser.add_argument('--split_ratio', type=float, default=0.33, help='Ratio for the test set, rest will become the train set')
# Variables for the configuration of the BERT model
parser.add_argument('--sequence_length', type=int, default=48, help='Sequence length to be used in the model')
parser.add_argument('--hidden_size', type=int, default=192, help='Hidden dimension size for the encoders in BERT')
parser.add_argument('--num_hidden_layers', type=int, default=6, help='Number of encoder layers in BERT')
parser.add_argument('--num_attention_heads', type=int, default=6, help='Number of attention heads in BERT')
# Variables for the optimization algorithm
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999), help='beta1 and beta2 parameters for the AdamW optimizer')
parser.add_argument('--eps', type=float, default=1e-8, help='Epsilon value to be used for optimization')
parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimization')
parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train each model')
parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for optimization')
# Variables for configuration of the training tasks
parser.add_argument('--mask_probability', type=float, default=0.25, help='Probability of [MASK]ing each token for pretraining')
parser.add_argument('--pretrain', default=False, action='store_true', help='Specify for pretraining (MLM), else only finetuning (classification)')
parser.add_argument('--save_models', default=False, action='store_true', help='Specify to save models (same keyword for pretraining and finetuning)')
args = parser.parse_args()
print('ARGS: ', args)

# Configure device
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('DEVICE: ', DEVICE)


# TODO: This sequence length might be need to be updated -- we are not counting [SEP], [CLS], etc.

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
                          model_max_length=args.sequence_length,
                          do_lower_case=False,  # NOTE: Don't lower-case for T and F
                          do_basic_tokenize=True,
                          unk_token='[UNK]',
                          sep_token='[SEP]',
                          pad_token='[PAD]',
                          cls_token='[CLS]',
                          mask_token='[MASK]')

if args.pretrain:
    # Get (self-supervised) datasets and data collator for masked-language-modeling
    train_dataset = LogicDataset(data_path=DATA_PATH,
                                 tokenizer=TOKENIZER,
                                 sequence_length=args.sequence_length,
                                 split='train',
                                 split_ratio=args.split_ratio,
                                 supervision_mode='self_supervised',
                                 data_mode='pairs',
                                 seed=args.seed)
    test_dataset = LogicDataset(data_path=DATA_PATH,
                                tokenizer=TOKENIZER,
                                sequence_length=args.sequence_length,
                                split='test',
                                split_ratio=args.split_ratio,
                                supervision_mode='self_supervised',
                                data_mode='pairs',
                                seed=args.seed)
    data_collator = LogicMLMDataCollator(tokenizer=TOKENIZER, mask_probability=args.mask_probability)
    print('Num Training Examples: %d \t Num. Testing Examples: %d' % (len(train_dataset), len(test_dataset)))

    # Get dataloaders
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              collate_fn=data_collator.collate_batch,
                              pin_memory=True,
                              shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             collate_fn=data_collator.collate_batch,
                             pin_memory=True,
                             shuffle=False)

    # Configure our own custom BERT
    config = BertConfig(vocab_size=VOCABULARY_SIZE,
                        hidden_size=args.hidden_size,
                        num_hidden_layers=args.num_hidden_layers,
                        num_attention_heads=args.num_attention_heads,
                        max_position_embeddings=args.sequence_length)
    model = BertForMaskedLM(config=config)
    print('BERT Num. Parameters: %d' % model.num_parameters())
    # Place model on device
    model = model.to(DEVICE)

    # Initialize optimizer and scheduler for MLM task
    optimizer, scheduler = get_optimizer_and_scheduler(model=model,
                                                       learning_rate=args.learning_rate,
                                                       betas=args.betas,
                                                       weight_decay=args.weight_decay,
                                                       eps=args.eps,
                                                       num_warmup_steps=50,
                                                       num_training_steps=int(args.num_epochs*len(train_dataset)/args.batch_size))

    print('--------------PRETRAINING----------------')
    # (1) PRETRAINING ON MASKED LANGUAGE MODELING
    # Start actual training, check test loss after each epoch
    best_test_loss = float('inf')
    for epoch in range(args.num_epochs):
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

        if args.save_models and test_loss < best_test_loss:
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
                             sequence_length=args.sequence_length,
                             split='train',
                             split_ratio=args.split_ratio,
                             supervision_mode='supervised',
                             data_mode='pairs',
                             seed=args.seed)
test_dataset = LogicDataset(data_path=DATA_PATH,
                            tokenizer=TOKENIZER,
                            sequence_length=args.sequence_length,
                            split='test',
                            split_ratio=args.split_ratio,
                            supervision_mode='supervised',
                            data_mode='pairs',
                            seed=args.seed)
print('Num Training Examples: %d \t Num. Testing Examples: %d' % (len(train_dataset), len(test_dataset)))

# Get dataloaders
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=args.batch_size,
                          pin_memory=True,
                          shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=args.batch_size,
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
                                                   learning_rate=args.learning_rate,
                                                   betas=args.betas,
                                                   weight_decay=args.weight_decay,
                                                   eps=args.eps,
                                                   num_warmup_steps=50,
                                                   num_training_steps=int(args.num_epochs*len(train_dataset)/args.batch_size))

best_test_loss = float('inf')
for epoch in range(args.num_epochs):
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

    if args.save_models and test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save(model.state_dict(), os.path.join(FINETUNED_SAVE_DIR, 'step_classifier.pt'))

    print(f'\tTrain Loss: {train_loss:.3f} | Train Accuracy: {train_accuracy * 100:.2f}%')
    print(f'\tTest Loss:  {test_loss:.3f} | Test Accuracy:  {test_accuracy * 100:.2f}%')
