# document-level-mt-project



## Tasks

1. Dataset
    a. test, train, aligned split - DONE
    b. Sentence level document tokenize split - DONE
    c. Process Unaligned sentences (using sentence transformer) - LATER

2. Evaluation Metrics
    a. Baseline Results using these metrics (using base MarianMT)

3. Report

4. Full scale fine tuning
    a. Fine tune all layers
    b. reduced fine tuning (on some layers only)

5. Modified Self-training - without fine tuning, using basic self-training on base MarianMT
    a. Self-training with token truncation, grad-clipping (without freezing)
    b. Freeze layers - unfreeze last two layer of decoder (and encoder?)
    c. Hybrid loss self-training (using gold labels)

6. Other approaches
    a. LORA adapters

7. Others
    a. Dataset and Dataloaders
    b. 

## Assigneeeeeee

- Tanmay
    - Modified Self-training (b)

- Kyle
    - Evaluation Metrics
        - BLEU, ROGUE, etc
    - Baseline MarianMT model results using the above metrics

- Anish
    - Report
    - Modified Self-training (a)

- Vishal
    - Full-scale Fine-tuning of MarianMT on PM India dataset 
        - And evaluation results

- Mayur
    - 5.c
    - 6.a
        