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

<details>
  <summary><h3>0. Download data</h3></summary>

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

<details>
  <summary><h3>1. Creating knowledge conflict dataset</h3></summary>

  #### Stage 1: Closed-book answer gathering
  We run the closed-book experiments using configs in <code>config/cb</code>.

  ```
  python 1_gather_cb_answers.py --config config/cb/llama7b/hotpotqa.conf
  ```

  #### Stage 2: Filtering out no-conflict examples
  ```
  python 2_filter_out_no_conflict.py --config config/filter/llama7b/hotpotqa.conf 
  ```

</details>

<details>
  <summary><h3>2. Studying knowledge updating behaviors under knowledge conflict</h3></summary>

  #### Section 4.2 Studying knowledge updating behaviors under realistic knowledge conflicts
  In this experiment, we run stage 3 of the pipeline.
  We run the open-book experiments using configs in <code>config/ob</code>.
  By default, the results are saved into <code>results/<model_name>/ob _<dataset>.out</code>.
  

  ```
  python 3_run_ob_experiment.py --config config/ob/llama7b/hotpotqa.conf
  ```

</details> 

