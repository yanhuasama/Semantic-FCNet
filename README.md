# Semantic-FSNet

This repository contains the code for  Semantic-FSNet: Leveraging Contextual Semantic and Multi-Level Feature Aggregation for Few-Shot Learning.

## Main Results

#### 5-way accuracy (%) on *miniImageNet*

|     method     | 1-shot | 5-shot |
| :------------: | :----: | :----: |
|     SEVPro     | 71.81  | 78.88  |
|     MAVSI      | 69.74  | 82.23  |
|      LPE       | 71.64  | 79.67  |
| Semantic-FSNet | 74.89  | 79.60  |

#### 5-way accuracy (%) on *tieredImageNet*

|     method     | 1-shot | 5-shot |
| :------------: | :----: | :----: |
|     SEVPro     | 72.77  | 84.04  |
|     MAVSI      | 74.61  | 87.45  |
|      LPE       | 73.88  | 88.48  |
| Semantic-FSNet | 74.62  | 82.20  |

#### 5-way accuracy (%) on *Cifar-fs*

|     method     | 1-shot | 5-shot |
| :------------: | :----: | :----: |
|     SEVPro     | 80.36  | 86.12  |
|     MAVSI      | 80.12  | 87.13  |
|      LPE       | 80.62  | 86.22  |
| Semantic-FSNet | 82.93  | 87.49  |

## Running the code

### Preliminaries

**Environment**

- Python 3.7.3
- Pytorch 1.2.0
- tensorboardX

**Datasets**

- miniImageNet
- tieredImageNet
- Cifar-fs
- FC100

In following we take miniImageNet as an example. For other datasets, replace `mini` with `tiered`  or others.
By default it is 1-shot, modify `shot` in config file for other shots. Models are saved in `save/`.

### 1. Training Classifier-Baseline

```
python train_classifier.py --config configs/train_classifier_mini.yaml
```

### 2. Training Semantic-FSNet

```
python train_meta.py --config configs/train_meta_mini.yaml
```

### 3. Test

To test the performance, modify `configs/test_few_shot.yaml` by setting `load_encoder` to the saving file of Classifier-Baseline, or setting `load` to the saving file of Semantic-FSNet.

E.g., `load: ./save/meta_mini-imagenet-1shot_meta-baseline-resnet12/max-va.pth`

Then run

```
python test_few_shot.py --shot 1
```

## Advanced instructions

### Configs

A dataset/model is constructed by its name and args in a config file.

For a dataset, if `root_path` is not specified, it is `materials/{DATASET_NAME}` by default.

For a model, to load it from a specific saving file, change `load_encoder` or `load` to the corresponding path.
`load_encoder` refers to only loading its `.encoder` part.
