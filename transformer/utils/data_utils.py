import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset

from utils.logic_utils import load_pickle_data


class LogicDataset(Dataset):
    """
    A dataset for easily iterating over and performing common operations on logic trees where
    a single data example looks like:
    (('(T∨~p)', [1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0]),
     [('T', 0, [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
      ('(T∨~p)', 4, [1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0])])

    where the first element is the starting expression and its bag-of-words (BOW) representation,
    and the second element is a list of items where each item is a representation, its relation
    to the item that comes before it (e.g. 1 -> identity, 4 -> demorgan, etc.), and its BOW
    representation. The last item in the second element is always the same expression in first
    element.

    :param (str) data_path: path to a .pkl (Pickle) that contains the actual logic tree data
    :param (transformers.BertTokenizer): tokenizer with a encode() function to convert tokens to IDs
    :param (int) sequence_length: maximum length for the sequence
    :param (str) supervision_mode: 'self_supervised' for only returning X, or 'supervised' for
           returning X, Y
    :param (str) data_mode: 'pairs' for <expression1 [SEP] <expression2> where the two expressions
           are connected by a logic law (e.g. demorgan, associativity, etc.) or 'sequential' for
           the full left-to-right tree <expression1 [SEP] expression2 [SEP] expression 3 [SEP] ...>
    :param (str) split: set to 'train' to get the training dataset, or 'test' for testing dataset
    :param (float) split_ratio: ratio (out of 1.0) of examples to hold out for testing
    :param (list) specific_laws: in the case where @data_mode = 'pairs', this list of strings
           contains the names of the laws that should be included in the dataset; others are dropped
    :param (int) seed: random state to be used while splitting the data into train and test splits
    """
    def __init__(self, data_path, tokenizer, sequence_length,
                 data_mode, supervision_mode, tokenization_mode='bert',
                 split='train', split_ratio=0.33, specific_laws=None, seed=42):
        super(LogicDataset).__init__()

        self.tokenizer = tokenizer
        self.sequence_length = sequence_length

        if data_mode == 'sequential' and supervision_mode == 'supervised':
            raise ValueError('Labels for sequential logic trees are not prepared yet!')
        self.data_mode = data_mode
        self.supervision_mode = supervision_mode
        self.tokenization_mode = tokenization_mode

        assert data_path.endswith('.pkl')
        X, Y = load_pickle_data(data_path=data_path,
                                specific_laws=specific_laws,
                                data_mode=data_mode)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                            test_size=split_ratio,
                                                            random_state=seed)

        if split == 'train':
            self.examples, self.labels = X_train, Y_train
        elif split == 'test':
            self.examples, self.labels = X_test, Y_test
        else:
            raise ValueError('Only allowed split configs are "train" and "test"!')

        counts = np.bincount(self.labels)
        ii = np.nonzero(counts)[0]
        print('Label Distribution for %s: ' % split, list(zip(ii, counts[ii])))

    def get_num_labels(self):
        return len(np.unique(self.labels))

    def get_labels(self):
        return np.unique(self.labels)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        if index < self.__len__():
            # Flatten example into a string with two expressions separated by [SEP] token
            example, label = self.examples[index, :], self.labels[index]
            example = example.tolist()
            # Remove any None variables that was previously added to equalize example lengths
            example = [expression for expression in example if expression]
            example = ' [SEP] '.join([' '.join(expression) for expression in example])
            # NOTE: 'bert' worked with a string due to WordPiece tokenization, but
            #       'gpt2' needed to be fed in as a list
            if self.tokenization_mode == 'gpt2':
                example = example.split()
            # Tokenize the example
            tokenized_example = self.tokenizer.encode(text=example,
                                                      add_special_tokens=True,
                                                      max_length=self.sequence_length,
                                                      truncation=True,
                                                      pad_to_max_length=True)
            # NOTE: encode(add_special_tokens=True) puts a [SEP] at the end of the sequence as well
            #       for the case of 'bert', but not for 'gpt2'. The way we implemented masked
            #       data collator, as well as other features, account for this.

            # Convert to a tensor
            tokenized_example = torch.tensor(data=tokenized_example, dtype=torch.long)
            label = torch.tensor(data=label, dtype=torch.long)
        else:
            raise ValueError('Out of range index while accessing dataset')

        if self.supervision_mode == 'self_supervised':
            return tokenized_example
        elif self.supervision_mode == 'supervised':
            return tokenized_example, label
        else:
            raise ValueError('Supervision mode "%s" not recognized' % self.supervision_mode)


class LogicMLMDataCollator(object):
    """
    Data collator used for masked language modeling (MLM) with logic expressions. It collates
    batches of tensors, honoring their tokenizer's pad_token and preprocesses batches with choosing
    each token with a @mask_probability probability, and replaces 80% of these with [MASK],
    10% with a random token from the vocabulary, and leaves the 10% as is.

    :param (BertTokenizer) tokenizer: tokenizer for the logic expressions with encode() and decode()
    :param (float) mask_probability: Probability used for choosing whether a token is to be included
           or not in the MLM task.
    """
    def __init__(self, tokenizer, mask_probability):
        if tokenizer.mask_token is None:
            raise ValueError(' Tokenizer is missing the mask token (e.g. [MASK], <mask>, etc.)')

        self.tokenizer = tokenizer
        self.mask_probability = mask_probability

        self.mask_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        self.nomask_id = -100

    def collate_batch(self, examples):
        batch = torch.stack(examples, dim=0)
        input_ids, labels = self.mask_tokens(input_ids=batch)
        return input_ids, labels

    def mask_tokens(self, input_ids):
        """
        Prepares masked tokens input and label pairs for MLM: 80% MASK, 10% random, 10% original.
        The @size arguments in torch functions used here (e.g. torch.full(), torch.randint())
        make this function work with a batch of input IDs. Moreover, functions like
        get_special_tokens_mask work for a single input ID, but is called in a for loop over the
        batch.

        :param (torch.Tensor) input_ids: tensor with shape (B, S), contains sequences
        :return: (torch.Tensor, torch.Tensor) input_ids, labels
        """
        # In masked language modeling, the labels are the input itself (i.e. self-supervision)
        labels = input_ids.clone()

        # We sample a few tokens in each sequence for masked-LM training with mask probability
        probabilities = torch.full(size=labels.shape, fill_value=self.mask_probability)
        # Get masks for special tokens, iterating over each label (list of input IDs) in the batch
        special_tokens_mask = []
        for labels_ in labels.tolist():
            mask = self.tokenizer.get_special_tokens_mask(labels_, already_has_special_tokens=True)
            special_tokens_mask.append(mask)
        # Get a boolean vector where T is for indices where special tokens are, otherwise F
        special_indices = torch.tensor(data=special_tokens_mask, dtype=torch.bool)
        # Fill in special token indices with 0.0 - we don't want them masked
        probabilities.masked_fill_(special_indices, value=0.0)

        # If a padding token (e.g. [PAD], <pad>, etc.) exists in the tokenizer
        if self.tokenizer._pad_token is not None:
            # Get the padding indices in the input IDs
            padding_indices = labels.eq(self.tokenizer.pad_token_id)
            # Fill in padding indices with 0.0 - we don't want them masked
            probabilities.masked_fill_(mask=padding_indices, value=0.0)

        # If a separation token (e.g. [SEP], <sep>, etc.) exists in the tokenizer
        if self.tokenizer._sep_token is not None:
            # Get the separation indices in the input IDs
            sep_indices = labels.eq(self.tokenizer.sep_token_id)
            # Fill in separation indices with 0.0 - we don't want them masked
            probabilities.masked_fill_(mask=sep_indices, value=0.0)

        # Get masked indices with a Bernoulli distribution based on p = probabilities
        masked_indices = torch.bernoulli(probabilities).bool()
        # Set everything except the masked indices to some large, negative number
        labels[~masked_indices] = self.nomask_id

        # 80% of the time, we replace masked input tokens with the mask token
        replaced_indices = torch.bernoulli(torch.full(size=labels.shape, fill_value=0.8)).bool()
        replaced_indices = replaced_indices & masked_indices
        input_ids[replaced_indices] = self.mask_id

        # 10% of the time, we replace the remaining masked input tokens with random tokens
        # NOTE: Fill value is 0.5, but the remaining pct is %20, which makes 0.2 x 0.5 = 0.10
        randomized_indices = torch.bernoulli(torch.full(size=labels.shape, fill_value=0.5)).bool()
        randomized_indices = randomized_indices & masked_indices & ~replaced_indices
        random_tokens = torch.randint(high=len(self.tokenizer), size=labels.shape, dtype=torch.long)
        input_ids[randomized_indices] = random_tokens[randomized_indices]

        # 10% of the time, we keep the remaining masked input tokens unchanged
        return input_ids, labels

    def sequentially_mask_tokens(self, input_ids, exclude_tokens=None):
        """
        Prepares masked tokens input and label pairs for MLM: given an input IDs of shape
        (B=1, S) (NOTE: The batch size should be 1), it sequentially creates a list of length S'
        of masked tokens input and label pairs, masking one token at a single time, going L-to-R.

        :param (torch.Tensor) input_ids: tensor with shape (B=1, S), contains sequences
        :param (list) exclude_tokens: list of tokens that should be not be masked
        :return: (torch.Tensor, torch.Tensor) input_ids, labels
        """
        # This function only works with a single example batch at a time
        assert input_ids.shape[0] == 1
        # Get token IDs to be excluded
        exclude_ids = self.tokenizer.convert_tokens_to_ids(exclude_tokens) if exclude_tokens else []
        # In masked language modeling, the labels are the input itself (i.e. self-supervision)
        labels = input_ids.clone()

        # Get masks for special tokens by iterating over input IDs and
        special_tokens_mask = []
        # Get masks for special tokens, iterating over each label (list of input IDs) in the batch
        for labels_ in labels.tolist():
            mask = self.tokenizer.get_special_tokens_mask(labels_, already_has_special_tokens=True)
            special_tokens_mask.append(mask)
        # Get a boolean vector where T is for indices where special tokens are, otherwise F
        special_indices = torch.tensor(data=special_tokens_mask, dtype=torch.bool)
        # Set special token (e.g. [CLS], [SEP]) indices to some large, negative number
        labels[special_indices] = self.nomask_id

        # If a padding token (e.g. [PAD], <pad>, etc.) exists in the tokenizer
        if self.tokenizer._pad_token is not None:
            # Get the padding indices in the input IDs
            padding_indices = labels.eq(self.tokenizer.pad_token_id)
            # Set padding indices (e.g. [PAD]) to some large, negative number
            labels[padding_indices] = self.nomask_id

        # If a separation token (e.g. [SEP], <sep>, etc.) exists in the tokenizer
        if self.tokenizer._sep_token is not None:
            # Get the separation indices in the input IDs
            sep_indices = labels.eq(self.tokenizer.sep_token_id)
            # Fill in separation indices with 0.0 - we don't want them masked
            labels[sep_indices] = self.nomask_id

        # Create a list of input IDs where each next example masks the next available index
        batch_input_ids = []
        # NOTE: We are momentarily converting labels to a one-dimensional vector of the form (S),
        #       from the previous (B=1,S) so that we can iterate over the actual input IDs
        for index, input_id in enumerate(labels[0, :].tolist()):
            if input_id != self.nomask_id and input_id not in exclude_ids:
                current_input_ids = input_ids.clone()
                current_input_ids[:, index] = self.mask_id
                batch_input_ids.append(current_input_ids)

        # The labels for each example in the created batch will be the same (i.e. original example)
        return batch_input_ids, labels


def get_features(input_ids, tokenizer, device):
    """
    Function to get BERT-related features, and helps to build the total input representation.

    :param (Tensor) input_ids: the encoded integer indexes of a batch, with shape: (B, P)
    :param (transformers.BertTokenizer) tokenizer: tokenizer with pre-figured mappings
    :param (torch.device) device: 'cpu' or 'gpu', decides where to store the outputted tensors
    :return: (Tensor, Tensor) token_type_ids, attention_mask: features describe token type with
             a 0 for the first sentence and a 1 for the pair sentence; enable attention on a
             particular token with a 1 or disable it with a 0
    """
    token_type_ids, attention_mask = [], []

    # Iterate over batch
    for input_ids_example in input_ids:
        # Convert tensor to a 1D list
        input_ids_example = input_ids_example.squeeze().tolist()
        # Set example to whole input when batch size is 1
        if input_ids.shape[0] == 1:
            input_ids_example = input_ids.squeeze().tolist()
        # Get padding information
        padding_token_id = tokenizer.convert_tokens_to_ids('[PAD]')
        padding_length = input_ids_example.count(padding_token_id)
        text_length = len(input_ids_example) - padding_length

        # Get separation information for assigning 0 to first statement, and 1 to second statement
        sep_token_id = tokenizer.convert_tokens_to_ids('[SEP]')
        # TODO: Change these with tokenizer.sep_token_id

        # Get all occurrences of [SEP]
        sep_locs = [i for i, x in enumerate(input_ids_example) if x == sep_token_id]

        # Sanity check to make sure no inputs start with the [SEP] token
        # assert 0 not in sep_locs
        # NOTE: Due to randomized word policy in mask_tokens(), this is not guaranteed!

        # Get segment IDs -> all 0s for first expression until first [SEP], all 1s for the second
        # sequence until second [SEP], and continues so on
        token_type_ids_example = [0] * sep_locs[0]
        for i in range(1, len(sep_locs)):
            token_type_ids_example.extend([i] * (sep_locs[i] - sep_locs[i-1]))

        # Complete token type IDs to full length with the next-up segment ID
        extend_length = len(input_ids_example) - len(token_type_ids_example)
        # If the next-token for the last-[SEP] is [PAD], fill with the latest, max token type ID
        if input_ids_example[sep_locs[-1] + 1] == padding_token_id:
            token_type_ids_example.extend([max(token_type_ids_example)] * extend_length)
        # Otherwise, fill with a new token type ID that is +1 than the latest one
        else:
            token_type_ids_example.extend([max(token_type_ids_example)+1] * extend_length)

        # Get input mask -> 1 for real tokens, 0 for padding tokens
        attention_mask_example = ([1] * text_length) + ([0] * padding_length)

        # Check if features are in correct length
        assert len(token_type_ids_example) == len(input_ids_example)
        assert len(attention_mask_example) == len(input_ids_example)
        token_type_ids.append(token_type_ids_example)
        attention_mask.append(attention_mask_example)

    # Convert lists to tensors
    token_type_ids = torch.tensor(data=token_type_ids, device=device)
    attention_mask = torch.tensor(data=attention_mask, device=device)
    return token_type_ids, attention_mask
