# DL4EEG-Classification 
These are the implementation of various deep learning based EEG classification models, including RGNN, DGCNN, BTA, HetEmotionNet, BENDR, EEGNet 

## Quick start
### Installing required packages
pip install -r requirements.txt

### Get the datasets
#### For our example dataset:
- The example dataset is sampled and preprocessed from the Search-Brainwave dataset. The example containing 10 folds. For each fold, there are 4 trainning samples and 1 testing sample.

#### For Search-Brainwave dataset:

- Download and preprocess acccording to the official code:

```
cd data_preprocess
python search_brainwave_preprocess.py
python search_brainwave_data_spliting.py
```

#### For AMIGOS dataset:

- download and preprocess:
```
cd data_preprocess
python amigos_preprocess.py
python amigos_data_spliting.py
```

### Running the model
- For BTA and example dataset:
```
cd scripts
sh bta_example.sh
```
OR
```
cd scripts
sh bta_example_unsupervised.sh
sh bta_example_supervised.sh
```

- For BTA and Search-Brainwave dataset:
```
cd scripts
sh main_unsupervised.sh
sh main.sh
```

- For BTA and AMIGOS dataset:
```
cd scripts
sh main_amigos_unsupervised.sh
sh main_amigos.sh
```

- For BENDR and example dataset
```
cd scripts
sh bendr_unsupervised.sh % loading the pretrained model weight
sh bendr_example.sh
```

- For EEGNet and example dataset
```
cd scripts
sh eegnet_example.sh
```

- For RGNN and example dataset
```
cd scripts
sh rgnn_example.sh
```

- For DGCNN and example dataset
```
cd scripts
sh dgcnn_example.sh
```

- For HetEmotionNet and example dataset
```
cd scripts
sh het_example.sh
```

## Dataset website
Search_Brainwave: http://www.thuir.cn/Search_Brainwave/

AMIGOS: http://www.eecs.qmul.ac.uk/mmv/datasets/amigos/index.html


## Reference

BTA: "Brain Topography Adaptive Network for Satisfaction Modeling in Interactive Information Access System".[unavailabled online]

RGNN: "EEG-based emotion recognition using regularized graph neural networks".[https://ieeexplore.ieee.org/abstract/document/9091308]

DGCNN: "EEG emotion recognition using dynamical graph convolutional neural networks".[https://ieeexplore.ieee.org/abstract/document/8320798]

HetEmotionNet: "HetEmotionNet: two-stream heterogeneous graph recurrent neural network for multi-modal emotion recognition".[https://dl.acm.org/doi/abs/10.1145/3474085.3475583]

BENDR: "BENDR: using transformers and a contrastive self-supervised learning task to learn from massive amounts of EEG data".[https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8261053/]

EEGNet: "EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces".[https://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta]

