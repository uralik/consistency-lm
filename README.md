
### I. Preparation

```bash
pip install -r requirements.txt
python setup.py develop
```


### II. Explanation of `run_slurm.py`

#### (i) Preliminaries

The authors used `run_slurm.py` to run experiments. 

The file contains six segments of code, with each segment headed by `if False`. The six segments correspond to the following (this information is also commented in `run_slurm.py`):
- (1) training vanilla models (RNNs and LSTMs)
- (2) evaluating vanilla model using regular decoding algorithms (for decoding algorithms, please refer to the decoding algorithm section below)
- (3) evaluating vanilla model using consistent sampling algorithms
- (4) training self-terminating RNN models
- (5) training self-terminating LSTM models
- (6) evaluating self-terminating RNN+LSTM models

Note that by default, we train each model using 10 different random seeds. The number of random seeds can be easily adjusted from `run_slurm.py`.

In the end of the segments we show how to modify parameters to run with BPE-tokenized dataset..

#### (ii) Things to do before using `run_slurm.py`

- Set the values in `user_folders` per the example in `run_slurm.py`
- Adjust `partition` choices in `run_slurm.py`, and adjust corresponding GPU options (to find the spot to do so, one can search `args.partition`)
- After training and before final evaluation, one should adjust `sweep_dirs` in the evaluation segments of the code, to refer to absolute locations of checkpoint folders; examples are included in `run_slurm.py`


#### (iii) How to use `run_slurm.py`

To use one segment, users can set the corresponding `if False` to `if True` and run `python run_slurm.py`. 

Alternatively, if users are not in a slurm environment, or if users prefer to run our code through command line, one can print out the actual python commands by including the flag `--print-commands`.



### III. Decoding algorithms in `evaluate.py`

- When the model is a self-terminating RNN/LSTM, `evaluate.py` only uses greedy decoding algorithm. 
- When the model is a regular RNN/LSTM...
  - if `--consistent-sampling 0`, then `evaluate.py` uses the following decoding algorithms: greedy decoding, ancestral sampling, beam search with beam size 2 and 4, top-k decoding with k=2 and k=4, and nucleus sampling with mu=0.2 and mu=0.4. 
  - if `--consistent-sampling 1`, then `evaluate.py` uses the following decoding algorithms: consistent top-k decoding with k=2 and k=4, and consistent nucleus sampling with mu=0.2 and mu=0.4. 


### IV. GPT-2 experiments

**The self-terminating wrapper supports Transformers 3.3.1 (current version on Oct 2020)**

The `gpt2` folder contains all the necessary wrappers to use self-terminating layer in HuggingFace pretrained model.

- (1) Tokenize wikitext-103 dataset: `prepare-wikitext.py`
- (2) Fine-tune GPT-2 or self-terminating GPT-2: `train_line.py` 
