# Introduction
This is the merge of two separate projects regarding egocentric action recognition with different goals.
Internally each of them has a README.
1. **EGO_Project_Group4** : Exploration of convolutional neural network trained end-to-end from raw data for predicting the action. In particular, I3D and TSM.
2. **AggregationMethods** : Exploration of temporal aggregation of the features extracted by the convolutional neural networks using average poolinand a temporal relation network TRN.
3. **MyTA3N** : Exploration of the domain adaptation task by using adversarial classifiers and its extension by introducing transformers.

# Environment


For running any experiment it is necessary to have a python environment properly configured.

`$python pip install -r requirements.txt`
## Dataset

The dataset it can be downloaded directly from the EPIC-KITCHENS website. Instructions can be found inside the 
**EGO_Project_Group4** folder. Features to be used with other networks can be extracted from them.