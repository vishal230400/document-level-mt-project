# Self-Training

## Setup
- Use python version 3.10.xx 
    - I tried with 3.13.xx and the tokenizer was breaking
- Install basic pytorch

- For dataset, download the `pmindia.v1.en.tgz` and `pmindia.v1.hi.tgz`, unzip them to get the documents in a folder named `split`.
    - Place `split` inside the `./dataset` folder
    - `./dataset/test1` folder contains handpicked documents with shorted sentences for testing model outputs.


## Self-Training

- `get_model()`
    - Freezes all layers except the last two decoder blocks

- `self_training()`
    - Basic Self training algorithm
    
    - Self Training Parameters (Below trend is based on 3-4 hours of testing only, feel free to try out other combinations)
        - lr: Set a lower lr (higher lr is too unstable)
        - decay_lambda: A lower decay value is halucinating the model even on self-training on the test set. (Higher lamda preferred)
        - num_steps and passes: Larger step-size and passes tends to overfit on a single sentence.



## Next Experiments to try

1. Try other combinations of parameter fine tuning
    1. Reduced Parameter Tuning
        - Un-freeze lesser decoder blocks?
        - Un-freeze few layer of encoder blocks also?
    2. Parameter Efficient Fine Tuning
        - Use adapaters (like LORA) - actually suggested by GPT for our current issues

2. Supervised Self-Training
    1. Use gold-labels as the pseudo-targets in the self training loop
    2. Modified loss function 
        - $loss = \alpha*\textit{generated} + (1-\alpha)*\textit{gold}$
    
3. Full-scale Fine Tuning
    1. Fully fine tune MarianMT on PM India Dataset
        - Supervised finetune using gold labels
        - Fine tune all layers? or reduced fine tuning?
