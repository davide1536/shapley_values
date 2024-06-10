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
![Alt](/results/waterfall_ensamble.png)
Waterfall graph that represents how different features "move" the prediction for a specific prediction x_i, from the expected value of all the prediction in the dataset.In red we have those features that increase the probability to survive, in blue those features that decrease the chance to survive.
![Alt](/results/ensamble_building3.png)
We have performed the ensemble building operation by constructing M' in a forward fashion, starting from the original ensemble M. In such operation we have used the shapley values as  metric to select this high performance subset.

As we can see in the above figure, the ensemble score tends to increase up to a certain number of sub-trees (80) and then it decreases. This is due to the fact that in the tail of the list $T$ we have trees with low shapley values and so they haven't got importance in the classification process or, they can misclassify some samples.

# Conclusion
the computation of the shapley values are useful in many different ways: starting from the mere interpretability of a machine learning model to the enhancing of its performance (ensemble building, feature analysis).
