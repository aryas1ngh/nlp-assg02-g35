# NLP CSE556 Assignment 1 - Group 35

## Build instructions
first activate your virtual environments (if any), for example:
```bash
source venv/bin/activate
```
or 
```bash
conda activate base
```

then download pretrained embeddings
```bash
chmod +x download_embeddings.sh
./download_embeddings.sh
```

then change directory to q1 (or whichever question you want to run) by doing
```bash
cd q1
```
and then run training experiments (glove, fasttext) embeddings X (1,2,3) layers ==> 6 variants
```bash
python3 q1.py --mode ablate
```
then test the model on the dummy `test_data.jsonl` file provided
```bash
python3 q1.py --mode test
```
please double check the dataset and other paths in the code before running the experiments