#!/bin/bash

# Function for cleaning the string
clean() {
    local a=${1//[^[:alnum:]]/}
    echo "${a,,}"
}


# Set global parameters
modality="RGB"


# Set file paths
source_domain=$1
target_domain=$2

if [ -z "$3" ]
then
  extra_parameters=""
else
  extra_parameters="$3"
fi





train_source_list="/home/andres/MLDL/EGO_Project_Group4/train_val/"$source_domain"_train.pkl"
train_target_list="/home/andres/MLDL/EGO_Project_Group4/train_val/"$target_domain"_train.pkl"
#shellcheck disable=SC2027
train_source_data="/home/andres/MLDL/Pre-extracted/"$modality"/ek_tsm/"$source_domain"-"$source_domain"_train"
#shellcheck disable=SC2027
train_target_data="/home/andres/MLDL/Pre-extracted/"$modality"/ek_tsm/"$source_domain"-"$target_domain"_train"



# Define the parameters we want to change

frame_aggregation=("avgpool" "trn-m")

# Define all the configurations to test
use_attn_and_place_adv=("none Y N N" "none N N Y"  \
"none N Y N" "none Y Y Y" "TransAttn Y Y Y")

#use_attn_and_place_adv=("none N N N")
## Do the loop in allfor frame_aggregation_value in ${frame_aggregation[*]}
for frame_aggregation_value in ${frame_aggregation[*]}
do
  for use_attn_and_place_adv_value in "${use_attn_and_place_adv[@]}" ; do

  # Get the attention and adversarial
  stringarray=($use_attn_and_place_adv_value)
val_list="str"
val_data="str"
echo main.py 8,8 $modality $train_source_list $train_target_list "$val_list" "$val_data" $train_source_data $train_target_data \
--num_segments 5 --baseline_type video --optimizer SGD --adv_DA RevGrad --train_metric verb \
--use_target uSv --frame_aggregation $frame_aggregation_value  --use_attn ${stringarray[0]} --place_adv ${stringarray[1]} ${stringarray[2]} ${stringarray[3]} \
--exp_path ./RESULTS$(clean "$extra_parameters")/$frame_aggregation_value${stringarray[0]}$(clean ${stringarray[1]})$(clean ${stringarray[2]})$(clean ${stringarray[3]})$source_domain$target_domain $extra_parameters

python main.py 8,8 $modality $train_source_list $train_target_list "$val_list" "$val_data" $train_source_data $train_target_data \
--num_segments 5 --baseline_type video --optimizer SGD --adv_DA RevGrad --train_metric verb \
--use_target uSv --frame_aggregation $frame_aggregation_value  --use_attn ${stringarray[0]} --place_adv ${stringarray[1]} ${stringarray[2]} ${stringarray[3]} \
--exp_path ./RESULTS$(clean "$extra_parameters")/$frame_aggregation_value${stringarray[0]}$(clean ${stringarray[1]})$(clean ${stringarray[2]})$(clean ${stringarray[3]})$source_domain$target_domain $extra_parameters


done
done








