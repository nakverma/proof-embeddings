import os
from tqdm import tqdm
import pandas as pd
import pickle
from collections import defaultdict
import argparse

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pylmnn import LargeMarginNearestNeighbor
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support, classification_report

import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import DataLoader, ConcatDataset

from transformers import BertTokenizer, BertConfig, BertForMaskedLM, BertForSequenceClassification

from utils.data_utils import LogicDataset, LogicMLMDataCollator
from utils.logic_utils import LogicLawCodes
from utils.model_utils import knn_classifier_evaluation, get_features, get_normalized_attention, PerformanceMetrics

# Variables for data ops.
parser = argparse.ArgumentParser(description='Script to test a BERT model on logic data')
parser.add_argument('--seed', type=int, default=42, help='Set seeds for reproducibility')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size to be used in training and testing')
parser.add_argument('--split_ratio', type=float, default=0.33, help='Ratio for the test set, rest will become the train set')
parser.add_argument('--data_source', type=str, default='synthetic', choices=['synthetic', 'real'], help='Select which data to work with')
# Variables for the configuration of the BERT model
parser.add_argument('--sequence_length', type=int, default=48, help='Sequence length to be used in the model')
parser.add_argument('--hidden_size', type=int, default=192, help='Hidden dimension size for the encoders in BERT')
parser.add_argument('--num_hidden_layers', type=int, default=6, help='Number of encoder layers in BERT')
parser.add_argument('--num_attention_heads', type=int, default=6, help='Number of attention heads in BERT')
# Variables for configuration of the testing tasks and applications
parser.add_argument('--load_choice', type=str, default='finetuned', choices=['pretrained', 'finetuned'], help='Select which saved model to test')
parser.add_argument('--generate_data', default=False, action='store_true', help='Specify to generate and save data, possibly overwriting .npy files')
parser.add_argument('--feature_extraction_method', type=str, default='pooled', choices=['sequence', 'pooled'], help='Select which features to use')
parser.add_argument('--evaluate', default=False, action='store_true', help='Specify to evaluate the learned representations on a classification task')
parser.add_argument('--evaluation_method', default='knn', choices=['lmnn+knn', 'knn'], help='Select which method to evaluate by')
parser.add_argument('--n_neighbors', type=int, default=3, help='Num. neighbors to use in LMNN (if applicable) and KNN')
parser.add_argument('--density_analysis', default=False, action='store_true', help='Specify to perform density analysis in the learned representation space')
parser.add_argument('--use_pca', default=False, action='store_true', help='Specify to use PCA to project down; preferred when dimensionality is high (before t-SNE)')
parser.add_argument('--pca_n_components', type=int, default=50, help='Number of components (i.e. principal axes) to use in PCA when applicable')
parser.add_argument('--tsne_n_components', type=int, default=2, help='Number of components (i.e. low-dimension axes) to use in t-SNE')
parser.add_argument('--compute_attention', default=False, action='store_true', help='Specify to compute attentions')
parser.add_argument('--attention_method', type=str, default='last_layer_heads_average_kth_token', help='Method to compute attentions with')
parser.add_argument('--exclude_special_tokens', default=True, action='store_true', help='Specify to exclude [CLS] and [SEP] from attention computation')
parser.add_argument('--anomaly_detection_method', default=None, choices=['autoencoder', 'oneclasssvm', 'isolationforest'], help='Select which AD method to use')
parser.add_argument('--include_fallacies', default=False, action='store_true', help='Specify to add fallacy steps in the test set')
args = parser.parse_args()
print('ARGS: ', args)

# Configure device
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('DEVICE: ', DEVICE)

# Set seeds for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if DEVICE == torch.device('cuda'):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Configure load paths
PRETRAINED_MODEL_PATH = os.path.join('saved_models', 'pretrained')
FINETUNED_MODEL_PATH = os.path.join('saved_models', 'finetuned', 'step_classifier.pt')

# Configure paths to .txt files for vocabulary and get vocabulary size
VOCABULARY_PATH = os.path.join('data', 'vocabulary.txt')
VOCABULARY_SIZE = len(open(VOCABULARY_PATH, 'r').readlines())
# Configure path to .pkl file for synthetic logic tree data or .tsv for real student answers
if args.data_source == 'synthetic':
    DATA_PATH = os.path.join('data', 'T_flat_unigram_dataset.pkl')
elif args.data_source == 'real':
    DATA_PATH = os.path.join('data', 'student_answers.tsv')

# Configure the tokenizer
TOKENIZER = BertTokenizer(vocab_file=VOCABULARY_PATH,
                          model_max_length=args.sequence_length,
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
                             sequence_length=args.sequence_length,
                             split='train',
                             split_ratio=args.split_ratio,
                             supervision_mode='supervised',
                             data_mode='pairs',
                             data_source=args.data_source,
                             seed=args.seed,
                             logging=True)
test_dataset = LogicDataset(data_path=DATA_PATH,
                            tokenizer=TOKENIZER,
                            sequence_length=args.sequence_length,
                            split='test',
                            split_ratio=args.split_ratio,
                            supervision_mode='supervised',
                            data_mode='pairs',
                            data_source=args.data_source,
                            seed=args.seed,
                            logging=True)
assert train_dataset.get_num_labels() == test_dataset.get_num_labels()

# Optionally append fallacies to the test set for the later analyses (e.g. anomaly detection)
if args.include_fallacies:
    if args.data_source == 'real':
        raise ValueError('You are adding synthetic fallacy data on top of real student data! Labels will not match!')

    fallacy_dataset = LogicDataset(data_path=os.path.join('data', 'F_flat_unigram_dataset.pkl'),
                                   tokenizer=TOKENIZER,
                                   sequence_length=args.sequence_length,
                                   split='test',
                                   split_ratio=args.split_ratio,
                                   supervision_mode='supervised',
                                   data_mode='pairs',
                                   specific_laws=['fallacy'],
                                   seed=args.seed)
    test_dataset = ConcatDataset([test_dataset, fallacy_dataset])

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

if args.load_choice == 'finetuned':
    # Configure our own custom BERT -- should be the same config as in train_language_model.py
    # NOTE: Don't forget to set output_hidden_states and output_attentions to True!
    config = BertConfig(vocab_size=VOCABULARY_SIZE,
                        hidden_size=args.hidden_size,
                        num_hidden_layers=args.num_hidden_layers,
                        num_attention_heads=args.num_attention_heads,
                        max_position_embeddings=args.sequence_length,
                        num_labels=train_dataset.get_num_labels(),
                        output_hidden_states=True,
                        output_attentions=True)

    model = BertForSequenceClassification(config=config)
    model.load_state_dict(torch.load(FINETUNED_MODEL_PATH, map_location=DEVICE))
elif args.load_choice == 'pretrained':
    model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_PATH,
                                                          output_hidden_states=True,
                                                          output_attentions=True,
                                                          num_labels=train_dataset.get_num_labels())
    # NOTE: In the case of pretrained, we won't be making use num_labels and hence the
    #       classification head on top of the base BERT model (i.e. encoder layers)

print('BERT Num. Parameters: %d' % model.num_parameters())
# Place model on device
model = model.to(DEVICE)

if args.generate_data:
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

        if args.feature_extraction_method == 'sequence':
            X_train.append(flattened_sequence_output)
        elif args.feature_extraction_method == 'pooled':
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

        if args.feature_extraction_method == 'sequence':
            X_test.append(flattened_sequence_output)
        elif args.feature_extraction_method == 'pooled':
            X_test.append(flattened_pooled_output)
        Y_test.append(y.data[0])

    # Convert to numpy array
    X_train, Y_train, X_test, Y_test = np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)
    # Sanity checks for data
    assert X_test.shape[0] == Y_test.shape[0] and X_train.shape[0] == Y_train.shape[0]
    assert X_train.shape[-1] == X_test.shape[-1]
    # Save the data for later use
    np.save(os.path.join('data', 'X_train_%s_%s.npy' % (args.feature_extraction_method, args.load_choice)), X_train)
    np.save(os.path.join('data', 'Y_train_%s_%s.npy' % (args.feature_extraction_method, args.load_choice)), Y_train)
    np.save(os.path.join('data', 'X_test_%s_%s.npy' % (args.feature_extraction_method, args.load_choice)), X_test)
    np.save(os.path.join('data', 'Y_test_%s_%s.npy' % (args.feature_extraction_method, args.load_choice)), Y_test)

if args.evaluate:
    X_train = np.load(os.path.join('data', 'X_train_%s_%s.npy' % (args.feature_extraction_method, args.load_choice)))
    Y_train = np.load(os.path.join('data', 'Y_train_%s_%s.npy' % (args.feature_extraction_method, args.load_choice)))
    X_test = np.load(os.path.join('data', 'X_test_%s_%s.npy' % (args.feature_extraction_method, args.load_choice)))
    Y_test = np.load(os.path.join('data', 'Y_test_%s_%s.npy' % (args.feature_extraction_method, args.load_choice)))

    # Sanity checks for data
    assert X_train.shape[0] == Y_train.shape[0] and X_test.shape[0] == Y_test.shape[0]
    assert X_train.shape[-1] == X_test.shape[-1]

    print('----------------EVALUATING-------------------')
    if args.evaluation_method == 'lmnn+knn':
        lmnn = LargeMarginNearestNeighbor(n_neighbors=args.n_neighbors, max_iter=50,
                                          n_components=X_train.shape[-1], verbose=1,
                                          random_state=args.seed, n_jobs=-1)
        lmnn.fit(X_train, Y_train)
        knn_classifier_evaluation(X_train=lmnn.transform(X_train), Y_train=Y_train,
                                  X_test=lmnn.transform(X_test), Y_test=Y_test,
                                  n_neighbors=args.n_neighbors)
    elif args.evaluation_method == 'knn':
        knn_classifier_evaluation(X_train=X_train, Y_train=Y_train,
                                  X_test=X_test, Y_test=Y_test,
                                  n_neighbors=args.n_neighbors)

if args.density_analysis:
    print('---------------DENSITY ANALYSIS----------------')
    X_test = np.load(os.path.join('data', 'X_test_%s_%s.npy' % (args.feature_extraction_method, args.load_choice)))
    Y_test = np.load(os.path.join('data', 'Y_test_%s_%s.npy' % (args.feature_extraction_method, args.load_choice)))

    test_df = pd.DataFrame({}, columns=['representation', 'label'])
    test_df['representation'] = X_test.tolist()
    test_df['label'] = Y_test.tolist()

    if args.use_pca:
        pca = PCA(n_components=args.pca_n_components, random_state=args.seed)
        X_test = pca.fit_transform(X_test)
        for i in range(args.pca_n_components):
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
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig(os.path.join('logs', 'Xtest_pca2d_%d_%s_%s.png' % (args.pca_n_components, args.feature_extraction_method, args.load_choice)))

    tsne = TSNE(n_components=args.tsne_n_components, verbose=1, perplexity=30.0, n_iter=1000, random_state=args.seed)
    X_test = tsne.fit_transform(X_test)
    for i in range(args.tsne_n_components):
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
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(os.path.join('logs', 'Xtest_tsne2d_%s_%s.png' % (args.feature_extraction_method, args.load_choice)))

if args.compute_attention:
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
                                                      method=args.attention_method,
                                                      k=raw.split().index('[SEP]'),
                                                      exclude_special_tokens=args.exclude_special_tokens,
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

if args.anomaly_detection_method:
    print('-----------------------ANOMALY DETECTION------------------------')
    if args.anomaly_detection_method == 'autoencoder':
        # NOTE: BERT could be viewed as an autoencoder (AE) language model
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
                    if law == 'fallacy' and loss > threshold:
                        TP += 1
                    elif law == 'fallacy' and loss <= threshold:
                        FP += 1
                    elif law != 'fallacy' and loss > threshold:
                        FN += 1
                    elif law != 'fallacy' and loss <= threshold:
                        TN += 1

            metrics = PerformanceMetrics(true_positive=TP, false_positive=FP, true_negative=TN, false_negative=FN)

            print('TP: %d \t FP: %d \t TN: %d \t FN: %d' %
                  (TP, FP, TN, FN))
            print('Accuracy: %0.4f \t Recall: %0.4f \t Precision: %0.4f \t F1 Score: %0.4f' %
                  (metrics.accuracy(), metrics.recall(), metrics.precision(), metrics.f1()))
            print('----------------------------------------------------------------------------------')

    elif args.anomaly_detection_method in ['oneclasssvm', 'isolationforest']:
        # Apply methodology only on test data to fairly assess capability of learned representations
        X_test = np.load(os.path.join('data', 'X_test_%s_%s.npy' % (args.feature_extraction_method, args.load_choice)))
        Y_test = np.load(os.path.join('data', 'Y_test_%s_%s.npy' % (args.feature_extraction_method, args.load_choice)))

        # Reformat the labels based on the definition by scikit-learn: 1 is for normal
        # data instances and -1 is for anomaly/outlier/novel data instances
        for law in LogicLawCodes.LAWS:
            if law['name'] == 'fallacy':
                Y_test[Y_test == law['label']] = -1
            else:
                Y_test[Y_test == law['label']] = 1

        # Define the model -- OneClassSVM() and IsolationForest() are commonly used
        # algorithms used for unsupervised outlier detection
        if args.anomaly_detection_method == 'oneclasssvm':
            clf = OneClassSVM(kernel='rbf', gamma='auto', verbose=True, random_state=args.seed)
        elif args.anomaly_detection_method == 'isolationforest':
            clf = IsolationForest(n_estimators=200, n_jobs=-1, random_state=args.seed, verbose=1)

        # Fit the model on the test set examples, without exposing the labels
        clf.fit(X_test)

        # Compute some metrics (positive class is 'fallacy' (i.e. anomaly class), and all other laws are negative class)
        TP, FP, TN, FN = 0, 0, 0, 0
        for ground_truth, prediction in zip(Y_test, clf.predict(X_test)):
            if prediction == -1 and ground_truth == -1:
                TP += 1
            elif prediction == -1 and ground_truth == 1:
                FP += 1
            elif prediction == 1 and ground_truth == 1:
                TN += 1
            elif prediction == 1 and ground_truth == -1:
                FN += 1

        metrics = PerformanceMetrics(true_positive=TP, false_positive=FP, true_negative=TN, false_negative=FN)

        print('TP: %d \t FP: %d \t TN: %d \t FN: %d' %
              (TP, FP, TN, FN))
        print('Accuracy: %0.4f \t Recall: %0.4f \t Precision: %0.4f \t F1 Score: %0.4f' %
              (metrics.accuracy(), metrics.recall(), metrics.precision(), metrics.f1()))
