import logging
import torch
from tqdm import tqdm
from utils.data_utils import get_features

from sklearn.neighbors import KNeighborsClassifier

from transformers import AdamW, get_linear_schedule_with_warmup


def knn_classifier_evaluation(X_train, Y_train, X_test, Y_test, n_neighbors):
    """
    Function to fit a K-Nearest Neighbors (KNN) algorithm to evaluate the learned
    representation space. The expectation is that same label instances will be located closer
    to each other in the feature space.
    """
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
    knn.fit(X_train, Y_train)
    train_acc, test_acc = knn.score(X_train, Y_train), knn.score(X_test, Y_test)
    print('Accuracy on train set of %d points: %0.4f' % (X_train.shape[0], train_acc))
    print('Accuracy on test set of %d points: %0.4f' % (X_test.shape[0], test_acc))


def accuracy(y_pred, y_true):
    """Function to calculate multiclass accuracy per batch"""
    y_pred_max = torch.argmax(y_pred, dim=-1)
    correct_pred = (y_pred_max == y_true).float()
    acc = correct_pred.sum() / len(correct_pred)
    return acc


def train(model, tokenizer, iterator, optimizer, scheduler,
          mode='masked_language_modeling', device='cpu'):
    """
    Function to carry out the training process.
    :param (torch.nn.Module) model: model object to be trained
    :param (BertTokenizer) tokenizer: tokenizer with encode() and decode()
    :param (torch.utils.data.DataLoader) iterator: data loader to iterate over batches
    :param (torch.optim.[...]) optimizer: optimization algorithm
    :param (torch.optim.[...]) scheduler: scheduler for warming up
    :param (str) mode: mode of the training, 'masked_language_modeling' or 'sequence_classification'
    :param (torch.device) device: 'cpu' or 'gpu', decides where to store the outputted tensors
    """
    model.train()
    epoch_loss, epoch_accuracy = 0.0, 0.0

    for batch in tqdm(iterator, desc='Iterating over Train Batches'):
        # Get training input IDs & labels from the current batch
        input_ids, labels = batch
        # Place examples & labels on device
        input_ids, labels = input_ids.to(device), labels.to(device)
        # Get corresponding additional features from the current batch
        token_type_ids, attention_mask = get_features(input_ids=input_ids,
                                                      tokenizer=tokenizer,
                                                      device=device)
        # Reset the gradients from previous processes
        optimizer.zero_grad()
        # Pass features through the model w/ or w/o BERT masks for attention & token type
        if mode == 'masked_language_modeling':
            predictions = model(input_ids=input_ids,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask,
                                masked_lm_labels=labels)
            # NOTE: MLM loss is returned directly by the model
            # TODO: Do this yourself!
            loss = predictions[0]
        elif mode == 'language_modeling':
            predictions = model(input_ids=input_ids,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask,
                                labels=input_ids)
            # NOTE: Labels **are shifted** inside the model, s.t. we directly set labels = input_ids
            # NOTE: LM loss is returned directly by the model
            # TODO: Do this yourself!
            loss = predictions[0]
        elif mode == 'sequence_classification':
            predictions = model(input_ids=input_ids,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask,
                                labels=labels)
            # NOTE: Classification loss is also returned directly by the model
            # TODO: Do this yourself!
            loss, logits = predictions[0], predictions[1]
            epoch_accuracy += accuracy(y_pred=logits, y_true=labels).item()

        else:
            raise ValueError('Mode "%s" is not recognized!' % mode)

        # Backward and optimize
        loss.backward()
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator), epoch_accuracy / len(iterator)


def test(model, tokenizer, iterator, mode='masked_language_modeling', device='cpu'):
    """
    Function to carry out the testing (or validation) process.
    :param (torch.nn.Module) model: model object to be trained
    :param (BertTokenizer) tokenizer: tokenizer with encode() and decode()
    :param (torch.utils.data.DataLoader) iterator: data loader to iterate over batches
    :param (str) mode: mode of the training, 'masked_language_modeling' or 'sequence_classification'
    :param (torch.device) device: 'cpu' or 'gpu', decides where to store the outputted tensors
    """
    model.eval()
    epoch_loss, epoch_accuracy = 0.0, 0.0

    with torch.no_grad():
        for batch in tqdm(iterator, desc='Iterating over Test Batches'):
            # Get testing input IDs & labels from the current batch
            input_ids, labels = batch
            # Place examples & labels on device
            input_ids, labels = input_ids.to(device), labels.to(device)
            # Get corresponding additional features from the current batch
            token_type_ids, attention_mask = get_features(input_ids=input_ids,
                                                          tokenizer=tokenizer,
                                                          device=device)
            # Pass features through the model w/ or w/o BERT masks for attention & token type
            if mode == 'masked_language_modeling':
                predictions = model(input_ids=input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask,
                                    masked_lm_labels=labels)
                # NOTE: MLM loss is returned directly by the model
                # TODO: Do this yourself!
                loss = predictions[0]
            elif mode == 'language_modeling':
                predictions = model(input_ids=input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask,
                                    labels=input_ids)
                # NOTE: Labels are shifted inside the model, s.t. we directly set labels = input_ids
                # NOTE: LM loss is returned directly by the model
                # TODO: Do this yourself!
                loss = predictions[0]
            elif mode == 'sequence_classification':
                predictions = model(input_ids=input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask,
                                    labels=labels)
                # NOTE: Classification loss is also returned directly by the model
                # TODO: Do this yourself!
                loss, logits = predictions[0], predictions[1]
                epoch_accuracy += accuracy(y_pred=logits, y_true=labels).item()
            else:
                raise ValueError('Mode "%s" is not recognized!' % mode)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator), epoch_accuracy / len(iterator)


def get_optimizer_and_scheduler(model, learning_rate, betas, weight_decay, eps,
                                num_warmup_steps, num_training_steps):
    # Define IDs and group model parameters accordingly
    no_weight_decay_identifiers = ['bias', 'LayerNorm.weight']
    grouped_model_parameters = [
        {'params': [param for name, param in model.named_parameters()
                    if not any(identifier_ in name for identifier_ in no_weight_decay_identifiers)],
         'lr': learning_rate,
         'betas': betas,
         'weight_decay': weight_decay,
         'eps': eps},
        {'params': [param for name, param in model.named_parameters()
                    if any(identifier_ in name for identifier_ in no_weight_decay_identifiers)],
         'lr': learning_rate,
         'betas': betas,
         'weight_decay': 0.0,
         'eps': eps},
    ]
    optimizer = AdamW(grouped_model_parameters)
    # NOTE: Warm-up steps are updates with low learning rates at the beginning of training.
    # After this warm-up, the regular learning rate (schedule) trains your model to convergence.
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)

    return optimizer, scheduler


def get_attention_nth_layer_mth_head_kth_token(attention_outputs, n, m, k, average_heads=False):
    """
    Function to compute attention weights by:
    i)   Take the attention weights from the nth multi-head attention layer assigned to kth token
    ii)  Take the mth attention head
    """
    if average_heads is True and m is not None:
        logging.warning("Argument passed for param @m will be ignored because of head averaging.")

    # Get the attention weights outputted by the nth layer
    attention_outputs_concatenated = torch.cat(attention_outputs, dim=0)       # (K, N, P, P)
    attention_outputs = attention_outputs_concatenated.data[n, :, :, :]        # (N, P, P)

    # Get the attention weights assigned to kth token
    attention_outputs = attention_outputs[:, k, :]                             # (N, P)

    # Compute the average attention weights across all attention heads
    if average_heads:
        attention_outputs = torch.sum(attention_outputs, dim=0)                # (P)
        num_attention_heads = attention_outputs_concatenated.shape[1]
        attention_outputs /= num_attention_heads
    # Get the attention weights of mth head
    else:
        attention_outputs = attention_outputs[m, :]                            # (P)

    return attention_outputs


def get_attention_average_first_layer(attention_outputs):
    """
    Function to compute attention weights by:
    i)   Take the attention weights from the first multi-head attention layer assigned to CLS
    ii)  Average each token across attention heads
    """
    return get_attention_nth_layer_mth_head_kth_token(attention_outputs=attention_outputs,
                                                      n=0, m=None, k=0,
                                                      average_heads=True)


def get_attention_average_last_layer(attention_outputs):
    """
    Function to compute attention weights by
    i)   Take the attention weights from the last multi-head attention layer assigned to CLS
    ii)  Average each token across attention heads
    """
    return get_attention_nth_layer_mth_head_kth_token(attention_outputs=attention_outputs,
                                                      n=-1, m=None, k=0,
                                                      average_heads=True)


def get_normalized_attention(model, tokenizer, raw_sentence, method='last_layer_heads_average',
                             n=None, m=None, k=None, exclude_special_tokens=True,
                             normalization_method='normal', device='cpu'):
    """
    Function to get the normalized version of the attention output of a FineTunedBert() model
    @param (torch.nn.Module) model: FineTunedBert() model to visualize attention weights on
    @param (str) raw_sentence: sentence in string format, preferably from the test distribution
    @param (str) method: method name specifying the attention output configuration, possible values
           are 'first_layer_heads_average', 'last_layer_heads_average', 'nth_layer_heads_average',
           'nth_layer_mth_head', and 'custom' (default: 'last_layer_heads_average')
    @param (int) n: layer no. (default: None)
    @param (int) m: head no. (default: None)
    @param (int) k: token no. (default: None)
    @param (bool) exclude_special_tokens: whether to exclude special tokens such as [CLS] and [SEP]
           from attention weights computation or not (default: True)
    @param (str) normalization_method: the normalization method to be applied on attention weights,
           possible values include 'min-max' and 'normal' (default: 'normal')
    @param (torch.device) device: 'cpu' or 'gpu', decides where to store the outputted tensors
    """
    if None in [n, m, k] and method == 'custom':
        raise ValueError("Must pass integer argument for params @n, @m, and @k " +
                         "if method is 'nth_layer_mth_head_kth_token'")
    elif None not in [n, m, k] and method != 'custom':
        logging.warning("Arguments passed for params @n, @m, or @k will be ignored. " +
                        "Specify @method as 'nth_layer_mth_head_kth_token' to make them effective.")

    # Plug in CLS & SEP special tokens for identification of start & end points of sequences
    if '[CLS]' not in raw_sentence and '[SEP]' not in raw_sentence:
        tokenized_text = ['[CLS]'] + tokenizer.tokenize(raw_sentence) + ['[SEP]']
    else:
        tokenized_text = tokenizer.tokenize(raw_sentence)

    # Call model evaluation as we don't want no gradient update
    model.eval()
    with torch.no_grad():
        # Create input IDs with batch size 1
        input_ids = torch.tensor(data=[tokenizer.convert_tokens_to_ids(tokenized_text)])
        token_type_ids, attention_mask = get_features(input_ids=input_ids,
                                                      tokenizer=tokenizer,
                                                      device='cpu')

        x_ = model(input_ids=input_ids,
                   token_type_ids=token_type_ids,
                   attention_mask=attention_mask)

        attention_outputs = x_[2]  # ([K] x (B, N, P, P))

    attention_weights = None
    if method == 'first_layer_heads_average':
        attention_weights = get_attention_nth_layer_mth_head_kth_token(
            attention_outputs=attention_outputs,
            n=0, m=None, k=0,
            average_heads=True
        )
    elif method == 'last_layer_heads_average':
        attention_weights = get_attention_nth_layer_mth_head_kth_token(
            attention_outputs=attention_outputs,
            n=-1, m=None, k=0,
            average_heads=True
        )
    elif method == 'last_layer_heads_average_kth_token':
        attention_weights = get_attention_nth_layer_mth_head_kth_token(
            attention_outputs=attention_outputs,
            n=-1, m=None, k=k,
            average_heads=True
        )
    elif method == 'nth_layer_heads_average':
        attention_weights = get_attention_nth_layer_mth_head_kth_token(
            attention_outputs=attention_outputs,
            n=n, m=None, k=0,
            average_heads=True
        )
    elif method == 'nth_layer_mth_head':
        attention_weights = get_attention_nth_layer_mth_head_kth_token(
            attention_outputs=attention_outputs,
            n=n, m=m, k=0,
            average_heads=False
        )
    elif method == 'custom':
        attention_weights = get_attention_nth_layer_mth_head_kth_token(
            attention_outputs=attention_outputs,
            n=n, m=m, k=k,
            average_heads=False
        )

    # Remove the beginning [CLS] & ending [SEP] tokens for better intuition
    if exclude_special_tokens:
        tokenized_text, attention_weights = tokenized_text[1:-1], attention_weights[1:-1]

    # Apply normalization methods to attention weights
    # i)  Min-Max Normalization
    if normalization_method == 'min-max':
        max_weight, min_weight = attention_weights.max(), attention_weights.min()
        attention_weights = (attention_weights - min_weight) / (max_weight - min_weight)

    # ii) Z-Score Normalization
    elif normalization_method == 'normal':
        mu, std = attention_weights.mean(), attention_weights.std()
        attention_weights = (attention_weights - mu) / std

    # Convert tensor to NumPy array
    attention_weights = attention_weights.data

    tokens_and_weights = []
    for index, token in enumerate(tokenized_text):
        tokens_and_weights.append((token, attention_weights[index].item()))

    return tokens_and_weights


def get_delta_attention(tokens_and_weights_pre, tokens_and_weights_post):
    """Function to compute the delta (change) in scaled attention weights before & after"""
    tokens_and_weights_delta = []
    for i, token_and_weight in enumerate(tokens_and_weights_pre):
        token,  = token_and_weight[0],
        assert token == tokens_and_weights_post[i][0]

        pre_weight = token_and_weight[1]
        post_weight = tokens_and_weights_post[i][1]

        tokens_and_weights_delta.append((token, post_weight - pre_weight))

    return tokens_and_weights_delta


class PerformanceMetrics(object):
    def __init__(self, true_positive, false_positive, true_negative, false_negative):
        self.true_positive = true_positive if true_positive > 0 else 1e-15
        self.false_positive = false_positive
        self.true_negative = true_negative
        self.false_negative = false_negative

    def accuracy(self):
        numerator = self.true_positive + self.true_negative
        denominator = self.true_positive + self.true_negative + self.false_positive + self.false_negative
        return numerator / denominator

    def precision(self):
        numerator = self.true_positive
        denominator = self.true_positive + self.false_positive
        return numerator / denominator

    def recall(self):
        numerator = self.true_positive
        denominator = self.true_positive + self.false_negative
        return numerator / denominator

    def f1(self):
        numerator = 2 * self.precision() * self.recall()
        denominator = self.precision() + self.recall()
        return numerator / denominator
