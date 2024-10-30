
# Detail of the available arguments in the config file
## High level keys
|key|type|required|description|
|---|---|---|---|
|name|str|_train_|Name for the experiment, used in the folder name|
|description|str||Description of the experiment|
|device|str ("cpu" or "\*cuda\*")|(default cpu)|device to use torch with|
|nb_epochs|int|_train_|Number of epochs for training|
|batch_size|int ot Iter[int]|_train_|batch sizes (by mode or not)|
|optimizer|dict|_train_|Optimizer dictionary, see below|
|model|dict|_train_|Model dictionary, see below|
|dataset|dict|_train_|Dataset dictionary, see below|
|train_labels|dict|_train_|Dictionary for the labels and their respective losses, see below|
|test_labels|list of string|_evaluate_|List of the labels necessary for testing in the dataset|
|evaluation|dict|_evaluate_|Dictionary to describe the metrics, see below|
|projection|dict|_dim. red._|Dictionary to describe the projection method, see below|

## Optimizer dictionary
|key|type|required|Implemented|description|
|---|---|---|---|---|
|name|str|*|Adam|Name of the optimizer|
|args|dict|*||Arguments for the torch optimizer function|

Exemple: 
* **Adam**, args: `lr`, `weight_decay`

## Dataset dictionary
|key|type|required|Implemented|description|
|---|---|---|---|---|
|dataset type|str|*|HITS_2D, HITS_contrastive|Dataset type, which controls which Dataset class will be used|
|dataset path|str|*||Path to a file describing the dataset|
|contrastive file path|str|||Path to a file describing the positive groups of the dataset|
|args|dict|||Arguments for the Dataset class initialization|

Exemple:
* _type_: **HITS_2D** (*HITS_2D_Dataset* class), _data path_: path to a hdf5 file, _args_: `duration` (int), `initial_duration` (int), `multiple` (int), `remove_pos_U` (bool), `remove_class_U` (bool)
* _type_: **HITS_contrastive** (*HITS_contrastive_Dataset* class), _data path_: path to a hdf5 file, _contrastive file path_: path to a json file, _args_: `duration` (int), `initial_duration` (int), `multiple` (int), `remove_pos_U` (bool), `remove_class_U` (bool), `neg_intrasub` (bool), `neg_samepos` (bool), `nb_neg` (int), `nb_pos` (int)

## Model dictionary
|key|type|required|Implemented|description|
|---|---|---|---|---|
|name|str|*|ConvolutionalAE, ConvEncoder, ConvDecoder, ClassifConvAE, Triplet, Contrastive, TripletAE, ContrastiveAE|Model type, which controls which Module class will be used|
|args|dict|||Arguments for the Model class initialization|

Exemple:
* _type_: **ConvolutionalAE** (*ConvolutionalAE* class), _args_: `input_channels` (int, default 3), `latent_channels` (int, default 4), `k_size` (int, default 3)  
* _type_: **ConvEncoder** (*ConvolutionalEncoder* class), _args_: `input_channels` (int, default 3), `output_channels` (int, default 4), `k_size` (int, default 3)
* _type_: **ConvDecoder** (*ConvolutionalDecoder* class), _args_: `input_channels` (int, default 4), `output_channels` (int, default 3), `k_size` (int, default 3)
* _type_: **ClassifConvAE** (*ClassifConvAE* class), _args_: `input_shape` (tuple), `n_classes` (int), `input_channels` (int, default 3), `latent_channels` (int, default 4)
* _type_: **Triplet** (*TripletNet* class), _args_: `embedding_net` (torch.nn.Module)
* _type_: **Contrastive** (*ContrastiveNet* class), _args_: `embedding_net` (model dictionary)
* _type_: **TripletAE** (*TripletNetAE* class), _args_: `encoder` (model dictionary), `decoder` (model dictionary)
* _type_: **ContrastiveAE** (*ContrastiveNetAE* class), _args_: `encoder` (model dictionary), `decoder` (model dictionary)

## Train labels dictionary
|key|type|required|description|
|---|---|---|---|
|labels|list of str|*|Labels needed for the evaluation|
|loss|list of str|*|Loss used for training|
|weight|list of float|*|Each associated loss weight|
|prediction position|list of (float or list of float)|*|Each associated loss indice for the model output(s)|
|args|dict||Loss argument (`temperature` for contrastive, `margin` for triplet...)|


## Test labels dictionary
A list of the labels for the test dataset.

## Evaluation dictionary
The *key* is the data split on which the metric has to be applied: ALL, TRAIN or TEST.
The *value* is a dictionary:
"labels": list of labels
"metric": list of metrics

## Available labels
* unsupervised: the label is the same as the first element of the dataset item
* class: the label is an int of the HITS class (EG, ES or A)
* soft class: the label is a tensor with the soft label of the HITS class, size 3 (or 4 is undefined class is kept)
* hard pos: the label is an int of the HITS position in cardiac cycle (1, 2, 3, 4 or -1)
* soft pos: the label is a tensor with the soft label of the HITS position in cardiac cycle, size 5

## Available losses
Losses have to have a PyTorch implementation, for gradient to be computed.
* MSE: mean squared error ([MSELoss](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html))
* CE: soft cross entropy ([CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html))
* Triplet: implemented triplet loss ([TripletLoss](src/Utils/ContrastiveLoss.py))
* Contrastive: implemented contrastive loss inspired by Khosla2020 ([ContrastiveLoss](src/Utils/ContrastiveLoss.py))
* ContrastiveCosSim: implemented contrastive loss with cosine similarity ([ContrastiveLossCosSim](src/Utils/ContrastiveLoss.py))

## Available metrics
Metrics are from PyTorch, from scikit-learn or elsewhere.
* MSE: mean squared error ([MSELoss](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html))
* MCC: Matthews correlation coefficient (matthews_corrcoef from scikit-learn), with the evaluated class (mcc_pos, mcc_class)

Point cloud metrics, the metric name is parsed between the different values:
* Type
    * silhouette: [scikit-learn implementation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html) 
    * OR categoryContinuity: [ours](src/utils/metrics.py)
* Category
    * pos: category is the HITS position in cardiac cycle 
    * OR class: category is the HITS type
    * OR detailed: category is (position, HITS type)
* Space
    * 2D
    * OR latent

_example:_ 'silhouette_class_2D

## Available dimension reduction
* [TSNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html):
    Args: n_components, perplexity, early_exaggeration, learning_rate, max_iter, n_jobs..