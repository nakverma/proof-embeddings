# Transformers

This directory contains experiments with transformers on various tasks related to the proofs and 
logic laws, expressions, and proofs. All code written is an extension on HuggingFace's
`transformers` library. This directory has:

* `train_bert.py` and `test_bert.py` scripts for:
    * Training and/or evaluating `BERT` on:
        * *Self-supervised* **masked-language modeling (MLM)** (a.k.a. CLOZE) task on synthetic, 
        tautology-bounded logic expressions.
        * *Supervised* **step classification** task on synthetic, tautology-bounded logic expressions,
        where by looking at two sequential expressions, the model tries to predict the 
        law-relationship between them (e.g. DeMorgan's, etc.)
        * *Unsupervised* **anomaly detection** task with negative classes as tautology-bounded
        logic expressions whose representations are self-supervisedly learned by a model, and
        where positive or novelty classes are fallacy-bounded expressions where the only exercised
        law is fallacy.
    * Computing and visualizing attention mechanisms.
* `train_gpt2.py` and `test_gpt2.ipynb` script and notebook for training a full language model 
(as opposed to a masked-only-language model) on synthetic, tautology-bounded logic expressions, and 
then unconditionally generating sequential logic expressions in a left-to-right manner.
* `utils`: Shared files amongst all models and tasks
    * `model_utils.py`: Helper methods for training and evaluating any transformer model.
    * `logic_utils.py`: Helper methods for processing logic trees and reading files.
    * `data_utils.py`: Helper methods for processing string data and masking operations.
* `data`: Data files such as the vocabulary used for different models. Note that `.json` is used
for the `GPT2` model and the `.txt` is used for the `BERT` model. The `merges.txt` file is also
only necessary for the `GPT2` model.
* `saved_models`: The `.pt` saved model files themselves are not added due to size-constraints,
but this directory still includes `config.json` files to give a sense of the models used. Both
models are severely cut-down versions of their respective transformer models, mainly due to
prevent overfitting and reduce complexity for the time being.
* `logs`: The logs show results for different experiments!


    
