# Introduction
This project is aimed to tackle the problem of explainability for black-boxes machine learning models, especially  for ensemble model and for the features within a dataset.
In order to accomplish this task we are going to compute a metric for each sub-model that compose an ensemble (explainability for ensemble) or feature (explainability for feature) called shapley value, that is a game theory concept. We will see that this metric can be used for ensemble building, ensemble description or for feature analysis.
## Project Description

The project is devided into 2 section:
- Feature evaluation: The behavior of machine learning models in taking out their decisions is treated as black-box hindering the interpretability of their decisions. In this project we use shapley values to explain the behavior of a classifier model in discriminating if a person survived or not when the Titanic sunk.
- Ensamble evaluation:  We introduce ensemble games, a class of transferable utility cooperative games. Each classifier in the ensemble receives the data point, and they output, for each data point, a probability distribution over the classes. The final decision will be given by taking in consideration every single distribution. Using these values an oracle quantifies the worth of each model in the ensemble (shapley values).

## Goals
The main goals in this project are:
- Tackle model explainability
- Tackle ensamble explainability
- Develops an analytical method for ensamble building

# Implementation
This project is based on the work proposed by A [Rozemberczki and Sarkar(2021)](https://arxiv.org/abs/2101.02153) and [Haneen Alsuradi]([http://example.com "Title"](http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf)).
# Results
## Metrics obtained
# Conclusion
