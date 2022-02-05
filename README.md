# metadrop-pytorch

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/meta-dropout-learning-to-perturb-latent/meta-learning-on-omniglot-1-shot-20-way)](https://paperswithcode.com/sota/meta-learning-on-omniglot-1-shot-20-way?p=meta-dropout-learning-to-perturb-latent)

Unofficial PyTorch Implementation of "Meta Dropout: Learning to Perturb Latent Features for Generalization" (ICLR 2020)

To Run:
`python main.py --phi`

To Run MAML:
`python main.py`

To get the data:
`python data_haebom/get_data.py`

## Results

|       | Omni. 1shot (main paper)| Omni. 1shot (Ours)|
| ------| ---------------- | ---------------- |
| MAML | 95.23          | 96.76 |
| Meta-dropout | __96.63__ | __97.12__ |

See the [runs](https://wandb.ai/joeljosephjin/metadrop-pytorch) and [report](https://wandb.ai/joeljosephjin/metadrop-pytorch/reports/Metadrop-in-PyTorch-An-Evaluation--Vmlldzo5OTQwMjg) on the results.


## To Install
`conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch`

## To Do
- develop the 'minimal' branch
- set a smaller runtime that can still track performance, so as to develop/debug faster