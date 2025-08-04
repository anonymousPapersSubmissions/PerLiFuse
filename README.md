# PerLiFuse


**PerLiFuse** is a *Fake News Detector* algorithms based on PyTorch, is designed for detecting misinformation. PerLiFuse incorporates cross-modal residual connections and coherence-guided feature modulation, leverages Kumaraswamy reparameterization for low-variance training, and models rich frequency gates to prevent collapse of trivial fusion policies.


## Installation

PerLiFuse is available for **Python 3.8** and higher. 

**Make sure [PyTorch](https://pytorch.org/)(including torch and torchvision) are already installed.**

## Usage
- from source

```bash
git clone https://github.com/anonymousPapersSubmissions/PerLiFuse && cd PerLiFuse
pip install -r requirements.txt
run python main.py
```
Rememeber to create a folder to save the trained model "saved_models"



## Datasets
We evaluate PrLiFuse on three datasets (Twitter, Weibo and Fakeddit) datasets. We have shared the Twitter dataset to provide a minimum guide to running the repo. Please contact us via the comment section to provide you other datasets. 
