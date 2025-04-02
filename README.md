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
    b. Full scale fine tuning + hybrid loss self-training

7. Others
    a. Dataset and Dataloaders
    b. 

## Assigneeeeeee

- Tanmay
    - 1.a - DONE
    - 1.b - DONE
    - Modified Self-training (b)

- Kyle
    - Evaluation Metrics
        - BLEU, ROGUE, etc
    - Baseline MarianMT model results using the above metrics

- Anish
    - Report - DONE
    - Modified Self-training (a)

- Vishal
    - 4.a - Done
    - 6.b

- Mayur
    - Initial 5.a, 5.b - DONE
    - 5.c - DONE
    - 6.a
        