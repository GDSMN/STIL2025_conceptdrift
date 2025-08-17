# Description

This repository contains the dataset and code used in our study.

- [./models]: This folder contains the trained word2vec model.

- [main.ipynb]: Python notebook file with each step to process and test the data.

- [embeddings.py]: Contains the code to generate embeddings

- [experiments.py]: Contains the code to setup the experiments. Defaults to the setup used in our paper.

- [kernel_two_samples_test.py]: Contains Emanuele's implementation of the test. Available on https://github.com/emanuele/kernel_two_sample_test.git

- [utils.py]: Contains code used on all other files. Standardizes and loads the dataset.


## More Information About the Data

The original FakeRecogna2 corpus is available at https://huggingface.co/datasets/recogna-nlp/fakerecogna2-abstrativa

# BibTex Entry

@inproceedings{2025_guedes_conceptDrift_fakeNews_STIL,
    author = {Manuela Guedes Wanderley and Lucca Baptista Silva Ferraz and Tiago Agostinho Almeida and Renato Moraes Silva},
    title = {A Moving Target: Detecting Concept Drift in Brazilian Portuguese Fake News},
     booktitle={Proceedings of the 16th Symposium in Information and Human Language Technology (STIL'2025)}, 
     year={2025},
     month=oct,
     address = {Fortaleza, CE, Brazil},
     publisher={Association for Computational Linguistics},
    pages = {1--12}
}