# XNLIvar: Basque and Spanish variation-inclusive NLI

This repository contains the data and code used in the paper **Lost in Variation? Evaluating NLI Performance in Basque and Spanish Geographical Variants**
 
This paper evaluates the capacity of current language technologies to understand Basque and Spanish language varieties. We use NLI as a pivot task and introduce a novel, manually-curated parallel dataset in Basque and Spanish and their corresponding variants. Empirical analysis of comprehensive crosslingual and in-context learning experiments with, respectively, encoder-only and decoder-based Large Language Models (LLMs), reveals a performance drop when processing linguistic variations, with more pronounced effects observed in Basque. Error analysis indicates that lexical overlap plays no role, suggesting that linguistic variation represents the primary reason for the lower results. All data and code in this repository are public under Attribution-NonCommercial 4.0 International license. 

## Data

It introduces XNLIvar, a novel, manually-curated, variation inclusive NLI datasets in Basque and Spanish for NLI evaluation. The [data](https://github.com/hitz-zentroa/XNLIvar/tree/main/data) folder is structured in the following folders: 

- [Basque data](https://github.com/hitz-zentroa/XNLIvar/tree/main/data/eu): It provides the original XNLI test data, as well as the native and variation inclusive datasets.
- [Spanish data](https://github.com/hitz-zentroa/XNLIvar/tree/main/data/es): It provides de original XNLI test data, as well as the Basque native data translated into Spanish and the variation inclusive data. 
- [Translations](https://github.com/hitz-zentroa/XNLIvar/tree/main/data/translations): It provides the automatic translations of the native and variation inclusive datasets into English. 
- [ablation-eu](https://github.com/hitz-zentroa/XNLIvar/tree/main/data/ablation-eu): The datasets used in the ablation experiments for Basque.
- [ablation-es](https://github.com/hitz-zentroa/XNLIvar/tree/main/data/ablation-es): The datasets used in the ablation experiments for Spanish.

## Scripts

It provides scripts to fine-tune and evaluate discriminaitive models, as well as scripts to perform zero-shot prompting experiments with generative models. 


## Citation
The [paper](https://arxiv.org/abs/2506.15239) that explains the dataset and experiments can be cited as follows:
```
@inproceedings{bengoetxea-et-al-2025,,
    title = "Lost in Variation? Evaluating NLI Performance in Basque and Spanish Geographical Variants",
    author = "Bengoetxea, Jaione  and
      Gonzalez-Dios, Itziar  and
      Agerri, Rodrigo",
      year = "2025",
      url = "https://arxiv.org/abs/2506.15239"
}
```
