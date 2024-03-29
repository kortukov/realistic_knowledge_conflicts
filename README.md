# Studying Large Language Model Behaviors Under Realistic Knowledge Conflicts


## Getting started

### Clone this repo
```bash
git clone https://github.com/kortukov/realistic_knowledge_conflicts
cd realistic_knowledge_conflicts
```

### Install dependencies
```bash
conda create -n realistic_kc python=3.10
conda activate realistic_kc
pip install -r requirements.txt
```

## Reproducing experiments

### 0. Download data
<details>
  <summary>Instructions:</summary>
  
  ####  Test data 
  We download the MrQA validation split that is used as test data: 
  NQ, SQuAD, NewsQA, TriviaQA, SearchQA, HotpotQA.
  ```
  python 0_download_data.py --dataset-type test
  ```

  ####  ICL data 
  For ICL we use the train split of each dataset.
  We shuffle the original data and only save 10 examples.
  ```
  python 0_download_data.py --dataset-type icl
  ```
  

</details>  

