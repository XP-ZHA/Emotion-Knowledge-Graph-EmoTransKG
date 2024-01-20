## Requirements

- Python 3.8.0
- PyTorch 1.10.1+cu12.2
- Transformers 4.27.4



The code has been tested on Ubuntu 20.04 using a single GPU.
<br>

## EmoTransNet: Training EmoTransKG

1. Please download the two CER datasets and put them in the data folder. Here we utilize the data and codes from to pre-train a conversation discourse parser and use that parser to extract discourse graphs in the four ERC datasets. And we utilize the codes from [here](https://github.com/declare-lab/conv-emotion/tree/master/COSMIC) to extract utterance features.
2. Run our model:

```bash
python main.py --dataset IEMOCAP --lr 1e-4 --dropout 0.2 --batch_size 16 --gnn_layers 2
```
