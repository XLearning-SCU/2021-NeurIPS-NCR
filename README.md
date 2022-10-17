
PyTorch implementation for [Learning with Noisy Correspondence for Cross-modal Matching](https://proceedings.neurips.cc/paper/2021/file/f5e62af885293cf4d511ceef31e61c80-Paper.pdf) (NeurIPS 2021 Oral).

## Update

- 2022-10-17, We provide the image urls of CC152K from [Conceptual Captions](https://ai.google.com/research/ConceptualCaptions) (CC), which might be helpful to your research.

```

|-- cc152k
|   |-- dev_caps_img_urls.csv
|   |-- test_caps_img_urls.csv
|   `-- train_caps_img_urls.csv

```

Use [img2dataset](https://github.com/rom1504/img2dataset) to download images from the csv files. [More details](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/cc3m.md)


## Introduction

### NCR framework
<img src="https://github.com/XLearning-SCU/2021-NeurIPS-NCR/blob/main/framework.png"  width="860" height="268" />

## Requirements

- Python 3.7
- PyTorch ~1.7.1
- numpy
- scikit-learn
- Punkt Sentence Tokenizer:
  
```
import nltk
nltk.download()
> d punkt
```
  
## Datasets

### MS-COCO and Flickr30K
We follow [SCAN](https://github.com/kuanghuei/SCAN) to obtain image features and vocabularies.

### CC152K
We use a subset of [Conceptual Captions](https://ai.google.com/research/ConceptualCaptions) (CC), named CC152K. CC152K contains training 150,000 samples from the CC training split, 1,000 validation samples and 1,000 testing samples from the CC validation split. We follow the pre-processing step in [SCAN](https://github.com/kuanghuei/SCAN) to obtain the image features and vocabularies. 

[Download Dataset](https://ncr-paper.cdn.bcebos.com/data/NCR-data.tar)

## Training and Evaluation

### Training new models from scratch

Modify the ```data_path``` and ```vocab_path```, then train and evaluate the model(s):

```train

# CC152K
python ./NCR/run.py --gpu 0 --workers 2 --warmup_epoch 10 --data_name cc152k_precomp --data_path data_path --vocab_path vocab_path

# MS-COCO: noise_ratio = {0, 0.2, 0.5}
python ./NCR/run.py --gpu 0 --workers 2 --warmup_epoch 10 --data_name coco_precomp --num_epochs 20 --lr_update 10 --noise_ratio 0.2 --data_path data_path --vocab_path vocab_path

# Flickr30K: noise_ratio = {0, 0.2, 0.5}
python ./NCR/run.py --gpu 0 --workers 2 --warmup_epoch 5 --data_name f30k_precomp --noise_ratio 0.2 --data_path data_path --vocab_path vocab_path

```
It should train the model from scratch and evaluate the best model.

### Pre-trained models and evaluation
The pre-trained models are available here:

1. CC152K model [Download](https://ncr-paper.cdn.bcebos.com/models/ncr_cc152k_model_best.pth.tar)
2. MS-COCO 0% noise model [Download](https://ncr-paper.cdn.bcebos.com/models/ncr_coco_0_model_best.pth.tar)
3. MS-COCO 20% noise model [Download](https://ncr-paper.cdn.bcebos.com/models/ncr_coco_0.2_model_best.pth.tar)
4. MS-COCO 50% noise model [Download](https://ncr-paper.cdn.bcebos.com/models/ncr_coco_0.5_model_best.pth.tar)
5. F30K 0% noise model [Download](https://ncr-paper.cdn.bcebos.com/models/ncr_f30k_0_model_best.pth.tar)
6. F30K 20% noise model [Download](https://ncr-paper.cdn.bcebos.com/models/ncr_f30k_0.2_model_best.pth.tar)
7. F30K 50% noise model [Download](https://ncr-paper.cdn.bcebos.com/models/ncr_f30k_0.5_model_best.pth.tar)

Modify the ```model_path```, ```data_path```, ```vocab_path``` in the ```evaluation.py``` file. Then run ```evaluation.py```:
```
python ./NCR/evaluation.py
```
> Note that for MS-COCO, please set ```split``` to ```testall```, and ```fold5``` to ```false``` (5K evaluation) or ```true``` (Five-fold 1K evaluation).

### Experiment Results:
<img src="https://github.com/XLearning-SCU/2021-NeurIPS-NCR/blob/main/mscoco_flickr30k.png"  width="740" height="434" />
<img src="https://github.com/XLearning-SCU/2021-NeurIPS-NCR/blob/main/cc152k.png"  width="565" height="238" />


## Citation

If NCR is useful to your research, please cite the following paper:
```
@article{huang2021learning,
  title={Learning with Noisy Correspondence for Cross-modal Matching},
  author={Huang, Zhenyu and Niu, Guocheng and Liu, Xiao and Ding, Wenbiao and Xiao, Xinyan and Wu, Hua and Peng, Xi},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```

## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)

## Acknowledgements
The code is based on [SGRAF](https://github.com/Paranioar/SGRAF) and [SCAN](https://github.com/kuanghuei/SCAN) licensed under Apache 2.0.
