import os
from tqdm import tqdm
import pandas as pd
import pickle
from collections import defaultdict

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pylmnn import LargeMarginNearestNeighbor

import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import DataLoader, ConcatDataset

from transformers import BertTokenizer, BertConfig, BertForMaskedLM, BertForSequenceClassification

from utils.data_utils import LogicDataset, LogicMLMDataCollator
from utils.logic_utils import LogicLawCodes
from utils.model_utils import get_features, get_normalized_attention, PerformanceMetrics

# Configure device
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('DEVICE: ', DEVICE)

# Configure load paths and choice for which model to be used in analysis
PRETRAINED_MODEL_PATH = os.path.join('saved_models', 'pretrained')
FINETUNED_MODEL_PATH = os.path.join('saved_models', 'finetuned', 'step_classifier.pt')
LOAD_CHOICE = 'finetuned'

# Set hyperparameters
SPLIT_RATIO = 0.33
SEED = 42
SEQUENCE_LENGTH = 48
HIDDEN_SIZE = 192
NUM_HIDDEN_LAYERS = 6
NUM_ATTENTION_HEADS = 6

# Set the operations you want to perform to True
GENERATE_DATA = True
FEATURE_EXTRACTION_METHODOLOGY = 'sequence'

EVALUATE = True
EVALUATION_METHODOLOGY = 'knn'
N_NEIGHBOURS = 3

DIMENSION_ANALYSIS = True
USE_PCA = False  # When dimensionality is high, PCA before T-SNE is preferred
PCA_N_COMPONENTS = 50
NUM_TSNE_COMPONENTS = 2

COMPUTE_ATTENTION = False
ATTENTION_COMPUTATION_METHOD = 'last_layer_heads_average_kth_token'
EXCLUDE_SPECIAL_TOKENS = True  # exclude [CLS] and [SEP] tokens from attention computations

UNSUPERVISED_ANOMALY_DETECTION = False

# Sanity check
assert FEATURE_EXTRACTION_METHODOLOGY in ['pooled', 'sequence']
assert EVALUATION_METHODOLOGY in ['lmnn+knn', 'knn']

# Log for visibility
print('FEATURE EXTRACTION METHODOLOGY: %s' % FEATURE_EXTRACTION_METHODOLOGY)
print('LOAD CHOICE: %s' % LOAD_CHOICE)

# Configure paths to .txt files for vocabulary and get vocabulary size
VOCABULARY_PATH = os.path.join('data', 'vocabulary.txt')
VOCABULARY_SIZE = len(open(VOCABULARY_PATH, 'r').readlines())
# Configure path to .pkl file for actual logic tree data
DATA_PATH = os.path.join('data', 'T_flat_unigram_dataset.pkl')

# Configure the tokenizer
TOKENIZER = BertTokenizer(vocab_file=VOCABULARY_PATH,
                          model_max_length=SEQUENCE_LENGTH,
                          do_lower_case=False,  # NOTE: No lower-case for T and F
                          do_basic_tokenize=True,
                          unk_token='[UNK]',
                          sep_token='[SEP]',
                          pad_token='[PAD]',
                          cls_token='[CLS]',
                          mask_token='[MASK]')

# Get datasets
train_dataset = LogicDataset(data_path=DATA_PATH,
                             tokenizer=TOKENIZER,
                             sequence_length=SEQUENCE_LENGTH,
                             split='train',
                             split_ratio=SPLIT_RATIO,
                             supervision_mode='supervised',
                             data_mode='pairs',
                             seed=SEED,
                             logging=False)
test_dataset = LogicDataset(data_path=DATA_PATH,
                            tokenizer=TOKENIZER,
                            sequence_length=SEQUENCE_LENGTH,
                            split='test',
                            split_ratio=SPLIT_RATIO,
                            supervision_mode='supervised',
                            data_mode='pairs',
                            seed=SEED,
                            logging=False)
print('Num Training Examples: %d \t Num. Testing Examples: %d' % (len(train_dataset), len(test_dataset)))

# Get dataloaders
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=1,
                          pin_memory=True,
                          shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=1,
                         pin_memory=True,
                         shuffle=False)

# Load pretrained model with correct config - we have to add num. labels for classification task
assert train_dataset.get_num_labels() == test_dataset.get_num_labels()


def evaluate(X_train, Y_train, X_test, Y_test):
    # Fit the nearest neighbors classifier
    knn = KNeighborsClassifier(n_neighbors=N_NEIGHBOURS, n_jobs=-1)
    knn.fit(X_train, Y_train)

    acc = knn.score(X_train, Y_train)
    print('Accuracy on train set of {} points: {:.4f}'.format(X_train.shape[0], acc))
    acc = knn.score(X_test, Y_test)
    print('Accuracy on test set of {} points: {:.4f}'.format(X_test.shape[0], acc))


if LOAD_CHOICE == 'finetuned':
    # Configure our own custom BERT -- should be the same config as in train_language_model.py
    # NOTE: Don't forget to set output_hidden_states and output_attentions to True!
    config = BertConfig(vocab_size=VOCABULARY_SIZE,
                        hidden_size=HIDDEN_SIZE,
                        num_hidden_layers=NUM_HIDDEN_LAYERS,
                        num_attention_heads=NUM_ATTENTION_HEADS,
                        max_position_embeddings=SEQUENCE_LENGTH,
                        num_labels=train_dataset.get_num_labels(),
                        output_hidden_states=True,
                        output_attentions=True)

    model = BertForSequenceClassification(config=config)
    model.load_state_dict(torch.load(FINETUNED_MODEL_PATH, map_location=DEVICE))
elif LOAD_CHOICE == 'pretrained':
    model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_PATH,
                                                          output_hidden_states=True,
                                                          output_attentions=True,
                                                          num_labels=train_dataset.get_num_labels())
    # NOTE: In the case of pretrained, we won't be making use num_labels and hence the
    #       classification head on top of the base BERT model (i.e. encoder layers)
else:
    raise ValueError('Load choice "%s" is not recognized!' % LOAD_CHOICE)

print('BERT Num. Parameters: %d' % model.num_parameters())
# Place model on device
model = model.to(DEVICE)

if GENERATE_DATA:
    X_train, Y_train = [], []
    for batch in tqdm(train_loader, desc='Extracting Features for Training Data'):
        x, y = batch
        token_type_ids, attention_mask = get_features(input_ids=x,
                                                      tokenizer=TOKENIZER,
                                                      device=DEVICE)
        x_ = model.bert(input_ids=x,
                        token_type_ids=token_type_ids,
                        attention_mask=attention_mask)

        sequence_output = x_[0]              # (B, P, H)
        pooled_output = x_[1]                # (B, H)
        hidden_outputs = x_[2]               # ([K+1] x (B, P, H))
        attention_outputs = x_[3]            # ([K] x (B, N, P, P))

        flattened_sequence_output = sequence_output.view(-1).detach().numpy()
        flattened_pooled_output = pooled_output.view(-1).detach().numpy()

        if FEATURE_EXTRACTION_METHODOLOGY == 'sequence':
            X_train.append(flattened_sequence_output)
        elif FEATURE_EXTRACTION_METHODOLOGY == 'pooled':
            X_train.append(flattened_pooled_output)
        Y_train.append(y.data[0])

    X_test, Y_test = [], []
    for batch in tqdm(test_loader, desc='Extracting Features for Testing Data'):
        x, y = batch
        token_type_ids, attention_mask = get_features(input_ids=x,
                                                      tokenizer=TOKENIZER,
                                                      device=DEVICE)
        x_ = model.bert(input_ids=x,
                        token_type_ids=token_type_ids,
                        attention_mask=attention_mask)

        sequence_output = x_[0]              # (B, P, H)
        pooled_output = x_[1]                # (B, H)
        hidden_outputs = x_[2]               # ([K+1] x (B, P, H))
        attention_outputs = x_[3]            # ([K] x (B, N, P, P))

        flattened_sequence_output = sequence_output.view(-1).detach().numpy()
        flattened_pooled_output = pooled_output.view(-1).detach().numpy()

        if FEATURE_EXTRACTION_METHODOLOGY == 'sequence':
            X_test.append(flattened_sequence_output)
        elif FEATURE_EXTRACTION_METHODOLOGY == 'pooled':
            X_test.append(flattened_pooled_output)
        Y_test.append(y.data[0])

    # Convert to numpy array
    X_train, Y_train, X_test, Y_test = np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)
    np.save(os.path.join('data', 'X_train_%s_%s.npy' % (FEATURE_EXTRACTION_METHODOLOGY, LOAD_CHOICE)), X_train)
    np.save(os.path.join('data', 'Y_train_%s_%s.npy' % (FEATURE_EXTRACTION_METHODOLOGY, LOAD_CHOICE)), Y_train)
    np.save(os.path.join('data', 'X_test_%s_%s.npy' % (FEATURE_EXTRACTION_METHODOLOGY, LOAD_CHOICE)), X_test)
    np.save(os.path.join('data', 'Y_test_%s_%s.npy' % (FEATURE_EXTRACTION_METHODOLOGY, LOAD_CHOICE)), Y_test)

if EVALUATE:
    X_train = np.load(os.path.join('data', 'X_train_%s_%s.npy' % (FEATURE_EXTRACTION_METHODOLOGY, LOAD_CHOICE)))
    Y_train = np.load(os.path.join('data', 'Y_train_%s_%s.npy' % (FEATURE_EXTRACTION_METHODOLOGY, LOAD_CHOICE)))
    X_test = np.load(os.path.join('data', 'X_test_%s_%s.npy' % (FEATURE_EXTRACTION_METHODOLOGY, LOAD_CHOICE)))
    Y_test = np.load(os.path.join('data', 'Y_test_%s_%s.npy' % (FEATURE_EXTRACTION_METHODOLOGY, LOAD_CHOICE)))

    # Sanity checks for data
    assert X_train.shape[0] == Y_train.shape[0] and X_test.shape[0] == Y_test.shape[0]
    assert X_train.shape[-1] == X_test.shape[-1]

    print('----------------EVALUATING-------------------')
    if EVALUATION_METHODOLOGY == 'lmnn+knn':
        lmnn = LargeMarginNearestNeighbor(n_neighbors=N_NEIGHBOURS, max_iter=50, n_components=X_train.shape[-1], verbose=1, n_jobs=-1)
        lmnn.fit(X_train, Y_train)
        print(evaluate(X_train=lmnn.transform(X_train), Y_train=Y_train, X_test=lmnn.transform(X_test), Y_test=Y_test))
    elif EVALUATION_METHODOLOGY == 'knn':
        print(evaluate(X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test))

if DIMENSION_ANALYSIS:
    X_train = np.load(os.path.join('data', 'X_train_%s_%s.npy' % (FEATURE_EXTRACTION_METHODOLOGY, LOAD_CHOICE)))
    Y_train = np.load(os.path.join('data', 'Y_train_%s_%s.npy' % (FEATURE_EXTRACTION_METHODOLOGY, LOAD_CHOICE)))
    X_test = np.load(os.path.join('data', 'X_test_%s_%s.npy' % (FEATURE_EXTRACTION_METHODOLOGY, LOAD_CHOICE)))
    Y_test = np.load(os.path.join('data', 'Y_test_%s_%s.npy' % (FEATURE_EXTRACTION_METHODOLOGY, LOAD_CHOICE)))

    # Sanity checks for data
    assert X_train.shape[0] == Y_train.shape[0] and X_test.shape[0] == Y_test.shape[0]
    assert X_train.shape[-1] == X_test.shape[-1]

    test_df = pd.DataFrame({}, columns=['representation', 'label'])
    test_df['representation'] = X_test.tolist()
    test_df['label'] = Y_test.tolist()

    print('---------------DIMENSION ANALYSIS----------------')
    if USE_PCA:
        pca = PCA(n_components=PCA_N_COMPONENTS)
        X_test = pca.fit_transform(X_test)
        for i in range(PCA_N_COMPONENTS):
            test_df['pca-%d' % (i + 1)] = X_test[:, i]
        print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

        plt.figure(figsize=(16, 10))
        sns.scatterplot(
            x='pca-1',
            y='pca-2',
            hue='label',
            palette=sns.color_palette('hls', len(set(test_df['label'].values.tolist()))),
            data=test_df,
            legend='full',
            alpha=0.9
        )
        plt.savefig(os.path.join('logs', 'Xtest_pca2d_%d_%s_%s.png' % (PCA_N_COMPONENTS, FEATURE_EXTRACTION_METHODOLOGY, LOAD_CHOICE)))

    tsne = TSNE(n_components=NUM_TSNE_COMPONENTS, verbose=1, perplexity=40.0, n_iter=1000)
    X_test = tsne.fit_transform(X_test)
    for i in range(NUM_TSNE_COMPONENTS):
        test_df['tsne-%d' % (i + 1)] = X_test[:, i]

    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x='tsne-1',
        y='tsne-2',
        hue='label',
        palette=sns.color_palette('hls', len(set(test_df['label'].values.tolist()))),
        data=test_df,
        legend='full',
        alpha=0.9
    )
    plt.savefig(os.path.join('logs', 'Xtest_tsne2d_%s_%s.png' % (FEATURE_EXTRACTION_METHODOLOGY, LOAD_CHOICE)))

if COMPUTE_ATTENTION:
    # Put laws (i.e. labels) inside this dictionary
    law_attentions = defaultdict(lambda: defaultdict(float))
    law_token_counts = defaultdict(lambda: defaultdict(int))

    for batch in tqdm(test_loader, desc='Computing Attention for Test Examples'):
        x, y = batch
        # Work with B (batch size) = 1 for visualization
        assert x.shape[0] == 1

        # Get law name (e.g. DeMorgans)
        law = LogicLawCodes().label2name(label=y.item())
        # Get the raw logic expression with tokens (e.g. pvq) so that we now the actual characters
        raw = ' '.join(TOKENIZER.convert_ids_to_tokens(x[0]))

        # NOTE: We'll compute attentions of either [CLS] or [SEP] here for convention
        tokens_and_weights = get_normalized_attention(model=model,
                                                      tokenizer=TOKENIZER,
                                                      raw_sentence=raw,
                                                      method=ATTENTION_COMPUTATION_METHOD,
                                                      k=raw.split().index('[SEP]'),
                                                      exclude_special_tokens=EXCLUDE_SPECIAL_TOKENS,
                                                      normalization_method='min-max',
                                                      device=DEVICE)

        for token, weight in tokens_and_weights:
            law_attentions[law][token] += weight
            law_token_counts[law][token] += 1

    # Normalize attention based on character counts
    for law in law_attentions:
        for token in law_attentions[law]:
            law_attentions[law][token] /= law_token_counts[law][token]

    print('Law Attentions: ', law_attentions)

    # Save law attentions to a Pandas DataFrame and then to a CSV
    df = pd.DataFrame({}, columns=['Law'] + list(law_attentions[list(law_attentions.keys())[0]].keys()))
    for index, law in enumerate(law_attentions):
        row = [law]
        for token in df.columns[1:]:
            if token in law_attentions[law]:
                row.append(law_attentions[law][token])
            else:
                row.append(-1.0)  # token not found in test examples of the given law!
        df.loc[index] = row

    df = df.set_index('Law')
    df.to_csv(os.path.join('logs', 'law_attentions.csv'))

    # Get the top K columns (i.e. tokens) for each row (i.e. law)
    K = 5
    df_ = pd.DataFrame({column: df.T[column].nlargest(K).index.tolist() for column in df.T}).T
    print('---------Top K=%d Law Attentions:--------' % K)
    print(df_)


# Try to free some memory up!
del train_loader, train_dataset, test_loader, model

if UNSUPERVISED_ANOMALY_DETECTION:
    DATA_PATH = os.path.join('data', 'F_flat_unigram_dataset.pkl')
    fallacy_dataset = LogicDataset(data_path=DATA_PATH,
                                   tokenizer=TOKENIZER,
                                   sequence_length=SEQUENCE_LENGTH,
                                   split='test',
                                   split_ratio=SPLIT_RATIO,
                                   supervision_mode='supervised',
                                   data_mode='pairs',
                                   specific_laws=['fallacy'],
                                   seed=SEED)
    test_dataset = ConcatDataset([test_dataset, fallacy_dataset])

    # Get data loaders
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=1,
                             pin_memory=True,
                             shuffle=True)

    # Get data collator, we will use this separately for each example
    data_collator = LogicMLMDataCollator(tokenizer=TOKENIZER, mask_probability=0.0)

    # Get the MLM model -- we will use the loss on CLOZE task to detect anomaly
    model = BertForMaskedLM.from_pretrained(PRETRAINED_MODEL_PATH)
    print('BERT Num. Parameters: %d' % model.num_parameters())
    # Place model on device
    model = model.to(DEVICE)

    # Store laws (i.e. labels) losses inside a dictionary (i.e. averaged over all law examples)
    law_losses = defaultdict(list)

    for batch in tqdm(test_loader, desc='Unsupervised Anomaly Detection for Test Examples'):
        x, y = batch
        # Work with B (batch size) = 1 for visualization
        assert x.shape[0] == 1

        # Initialize reconstruction loss for the current example
        # NOTE: Reconstruction loss will be defined as the sum of losses in sequential CLOZE task
        # NOTE: backward() will not be called on this loss; it is simply used for anomaly detection
        reconstruction_loss = 0.0

        # Remove 'F' token for now; fallacy was not included in the train set!
        # TODO: Other strategies to try include i) replacing with [UNK] token, and ii)
        #       ii) completely removing the token at that timestep.
        x[x == TOKENIZER.convert_tokens_to_ids('F')] = TOKENIZER.convert_tokens_to_ids('T')

        # Create list of input IDs and labels to sum the loss over
        batch_input_ids, labels = data_collator.sequentially_mask_tokens(input_ids=x)

        for input_ids in batch_input_ids:
            # Place examples & labels on device
            input_ids, labels = input_ids.to(DEVICE), labels.to(DEVICE)
            # Get corresponding additional features from the current batch
            token_type_ids, attention_mask = get_features(input_ids=input_ids,
                                                          tokenizer=TOKENIZER,
                                                          device=DEVICE)
            predictions = model(input_ids=input_ids,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask,
                                masked_lm_labels=labels)
            # NOTE: MLM loss is returned directly by the model
            # TODO: Do this yourself!
            loss = predictions[0]
            reconstruction_loss += loss.item()

        # Normalize reconstruction loss by the num. sequential examples (i.e. num masked tokens)
        reconstruction_loss /= len(batch_input_ids)
        # Update dictionaries for losses and counts
        law_losses[LogicLawCodes().label2name(y.item())].append(reconstruction_loss)

    # Print statistics for each law losses list
    for law in law_losses:
        print('Law: ', law)
        print('--------------------------------')
        print('MEAN: ', np.mean(law_losses[law]))
        print('STD: ', np.std(law_losses[law]))
        print('MIN: ', np.min(law_losses[law]))
        print('MAX: ', np.max(law_losses[law]))
        print('--------------------------------')

    # Save law losses for inspection!
    with open('law_losses.pkl', 'wb') as f:
        pickle.dump(law_losses, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Compute accuracy for anomaly detection (i.e. fallacy vs. all) across different threshold
    min_loss = np.min(law_losses['fallacy']) / 2
    max_loss = min_loss + ((np.mean(law_losses['fallacy']) - np.min(law_losses['fallacy'])) / 2)
    thresholds, num_steps = [], 10
    for i in range(num_steps):
        thresholds.append(min_loss + (((max_loss - min_loss) / num_steps) * i))

    for threshold in thresholds:
        print('Evaluating Performance for Threshold: %0.4f' % threshold)
        print('----------------------------------------------------------------------------------')
        # Compute some metrics (positive class is 'fallacy', and all other laws are negative class)
        TP, FP, TN, FN = 0, 0, 0, 0

        for law, losses in law_losses.items():
            for loss in losses:
                if law == 'fallacy':
                    if loss > threshold:
                        TP += 1
                    else:
                        FP += 1
                else:
                    if loss > threshold:
                        FN += 1
                    else:
                        TN += 1

        metrics = PerformanceMetrics(true_positive=TP,
                                     false_positive=FP,
                                     true_negative=TN,
                                     false_negative=FN)

        print('TP: %d \t FP: %d \t TN: %d \t FN: %d' %
              (TP, FP, TN, FN))
        print('Accuracy: %0.4f \t Recall: %0.4f \t Precision: %0.4f \t F1 Score: %0.4f' %
              (metrics.accuracy(), metrics.recall(), metrics.precision(), metrics.f1()))
        print('----------------------------------------------------------------------------------')
