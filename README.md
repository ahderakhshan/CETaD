# CETaD: Contradiction Extraction from Textual Data using Siamese Network Embeddings


This is the implementation of the paper CETaD: Contradiction Extraction from Textual Data using Siamese Network Embeddings.


## Overview
![](./CEMAT.png)

### siamese network training

To train Siamese network use following command

```bash
python train_siamese_network.py \
    --model_name_or_path roberta-large \
    --data SNLI MNLI \
    --tokenizer_max_length 128 \
    --epochs 6 \
    --learning_rate 2e-5 \
    --no_unfreeze_layer 6 
```
more arguments to customize training siamese network and arguments demonstrations can be found in 
``train_siamese_network.py`` file.

### Compute PPMI
to computing PPMI for each term in training data you should use following command
```bash
python compute_ppmi.py \
  --data SNLI MNLI \
  --min_freq 10 \
  --output_path ./ppmi 
```
this will store ppmi.pkl and terms.pkl file in output_path which can be used in contradiction extraction ranking. Consider that the values of stored ppmi's are summed with 1 based on paper details.

### Evaluate By Trained Siamese Network
To evaluate based on first approach (by trained siamese network) use following command. Arguments demonstration can be found in code.
```bash
python evaluate_by_siames_network.py \
  --data_path your_data_path \
  --checkpoint_path your_checkpoint_path \
  --tokenizer_max_length 128 \
  --e_th 0.8 \
  --c_th 0.8 \
  --alpha 1\
  --beta 1 \
  --number_of_extractions 50  
```
### Evaluate By Fact Checking Dataset
To evaluate based on second approach (by a corpus constructed from a fact-checking dataset) use following command. Arguments demonstration can be found in code.
```bash
python evaluate_by_siames_network.py \
  --data_path your_data_path \
  --checkpoint_path your_checkpoint_path \
  --tokenizer_max_length 128 \
  --e_th 0.8 \
  --c_th 0.8 \
  --alpha 1\
  --beta 1 \
  --top_n 50
```

### Evaluate By LLM
To evaluate based on third approach (use LLM to determine whether two sentences are contradictory or not ) use following command. Arguments demonstration can be found in code.
```bash
python evaluate_by_siames_network.py \
  --data_path your_data_path \
  --checkpoint_path your_checkpoint_path \
  --tokenizer_max_length 128 \
  --e_th 0.8 \
  --c_th 0.8 \
  --alpha 1\
  --beta 1 \
  --top_n 50 \
  --llm_model gpt-4o-mini \
  --api_key your_api_key
```
### Bugs or questions?

If you have any questions related to the code or the paper, feel free to email Amirhossein Derakhshan (`am_derakhshan@comp.iust.ac.ir` or `ahderakhshan.ce@gmail.com`). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!
