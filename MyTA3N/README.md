# Domain Adapatation in Egocentric Action Recognition with transformers
This project proposes a newly based architecture for domain adaptation in the field of egocentric action recognition. It combines the network TA3N with Transformers to 
This project is based on the original TA3N-EPIC-KITCHENS [code](https://github.com/jonmun/EPIC-KITCHENS-100_UDA_TA3N)


---
## Environment

1. Install required dependencies 
`$ pip install -r requirements.txt`

2. Download dataset. The project makes use of pre-extracted features from convolutional neural networks. Those can be directly extracted or downloaded from this [link](https://drive.google.com/drive/folders/1qF-AuitdjFguIjbZNwPYDMDHxyN_RfJo?usp=sharing).

3. Run tests using the scripts.

## Scripts

### Setup
Before running any script it is necessary to setup the paths inside the same scripts with the local configuration. This can be setup in the script
TestAllModalities.sh.
The python script datatset.py has also to be modified to include the paths of the labels.

### Run
The provided scripts are useful for running the test under all configurations.

1. TestAllConfigurations.sh : It uses the specified configurations and test them for all domains.
2. TestAllDomains.sh : Test the specified configuration on all domains
3. TestAllModalities.sh : Test all domains with the configuration and domain given as input for all temporal aggregators available.


Running those scripts will generate folders with the results.

An example of a running command is : 

`python main.py --dropout_i 0.8 --dropout_v 0.8 --add_fc 1 --use_relation_self_attention True --use_frame_self_attention True
main.py 8,8 RGB /home/andres/MLDL/EGO_Project_Group4/train_val/D1_train.pkl /home/andres/MLDL/EGO_Project_Group4/train_val/D2_train.pkl str str /home/andres/MLDL/Pre-extracted/RGB/ek_tsm/D1-D1_train /home/andres/MLDL/Pre-extracted/RGB/ek_tsm/D1-D2_train --num_segments 5 --baseline_type video --optimizer SGD --adv_DA RevGrad --train_metric verb --use_target uSv --frame_aggregation avgpool --use_attn none --place_adv Y N N --exp_path ./RESULTSdropouti08dropoutv08addfc1userelationselfattentiontrueuseframeselfattentiontrue/avgpoolnoneynnD1D2 --dropout_i 0.8 --dropout_v 0.8 --add_fc 1 --use_relation_self_attention True --use_frame_self_attention True`


## Model

The proposed model is found inside the *models.py* script. It is used inside the *main.py* module.

## Authors

Project by : Andres Cardenas , Edgar Gaytan and Luca Scalenghe.