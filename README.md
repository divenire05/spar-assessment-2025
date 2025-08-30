# SPAR-Assessment-2025
Take-home task for the following SPAR project: "Exploring dangerous hidden reasoning capabilities of language models"

# Fine‑Tuning DistilGPT‑2 on Pirate Speak

## Overview
For this task, I fine‑tuned a Hugging Face transformer model on a small custom dataset to teach it a simple behavior change: translating English phrases into “pirate‑speak.” I used DistilGPT‑2 (a lightweight generative model) and ran experiments on a Runpod T4/Ada GPU instance to demonstrate the effect of fine‑tuning.

The project shows:
- Training dynamics via a loss curve.
- Before vs after qualitative changes in model outputs.
- Reflections on the different approaches I tried (epochs, dataset size, preprocessing tweaks).


## Model & Dataset
- Base model: `distilgpt2` (~82M params).  
- Dataset: Custom JSON file of English→Pirate pairs.  

Example:  
```json
{"prompt": "Translate to pirate: Hello, friend!", "response": "Ahoy, matey!"}
```

I iterated on dataset size:
- Started with ~20 examples (very limited signal).
- Expanded to ~250, then ~500 examples by synthesizing more pairs.
- Observed that quantity without consistency hurt performance (overfitting, gibberish repetition).
- Settled back to a cleaner set of ~250 consistent examples of “short + piratey” responses.


## Training Setup
- Framework: Hugging Face Transformers + Trainer.
- GPU: Runpod single GPU instance.
- Hyperparameters:
    - Learning rate: 2e-5
    - Batch size: 2–4
    - Epochs: initially 3, then 5–10 → best results around ~8–10 epochs
- Change in code:
    - Early runs trained on prompt+response as labels → model learned to echo input (“Translate to pirate...”) before outputting pirate text.
    - Fixed preprocessing so labels = pirate response only. This encouraged more direct mappings like ```"Hello, friend!" → "Ahoy, matey!".```


## Training Loss Curve
The training loss (as shown in the train_loss.png file attached in the repo) decreased steadily over steps, confirming the model was fitting the task.


## Evaluation Results
Comparison of model outputs before and after fine‑tuning can be found in the eval_results.json file attached in the repo.


### Observations
- Base model rambles in irrelevant blog/news style text. It has no notion of pirate‑speak.
- Fine‑tuned model:
    - Sometimes still echoes the prompt.
    - Shows signs of piratey lexicon (“mate!”, “friend!!”).
    - Not perfectly consistent, but qualitatively different from base GPT‑2 → shows pirate elements not seen before training.


## Iterative Stages & Lessons
1. Initial runs (tiny dataset, 3 epochs): Loss went down, but outputs unchanged — still base‑like.
2. Increased epochs (10–20): More pirate words appeared, but beyond ~10 epochs → overfitting + collapse (repetition: “heart heart heart”).
3. Expanded dataset (500 examples): Made things worse when examples were noisy/inconsistent. Highlighted that data quality > data quantity.
4. Code fix (labels = response only): Prevented model from redundantly echoing ```"Translate to pirate:"```. Helped pirate outputs emerge.
5. Final approach: ~250 clean examples, 8–10 epochs, response‑only labels gave the clearest behavior change with outputs like "mate!" or the double exclamation following the "friend".


## Conclusion
Even with very small custom data, finetuning DistilGPT‑2 on Runpod demonstrated a visible behavior shift: the pretrained model rambled, while the finetuned model began producing pirate‑themed tokens.


## Repo Contents
- ```train.py```: fine‑tuning script.
- ```evaluate.py```: run prompts on base vs fine‑tuned model, outputs to JSON + Markdown.
- ```dataset/```: toy pirate dataset.
- ```train_loss.png```: loss curve plot.
- ```eval_results.json / eval_results.md```: saved evaluations.
- ```README.md```: this write‑up.

&nbsp;
### ended up spending less than $1   &ensp;    :smile: