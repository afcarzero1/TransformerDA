#!/bin/bash


configurations=(\
"--dropout_i 0.8 --dropout_v 0.8 --add_fc 1" "--dropout_i 0.8 --dropout_v 0.8 --add_fc 2" "--dropout_i 0.8 --dropout_v 0.8 --add_fc 3" \
"--dropout_i 0.8 --dropout_v 0.9 --add_fc 1" "--dropout_i 0.8 --dropout_v 0.9 --add_fc 2" "--dropout_i 0.8 --dropout_v 0.9 --add_fc 3" \
"--dropout_i 0.9 --dropout_v 0.9 --add_fc 1" "--dropout_i 0.9 --dropout_v 0.9 --add_fc 2" "--dropout_i 0.9 --dropout_v 0.9 --add_fc 3" \
"--dropout_i 0.9 --dropout_v 0.8 --add_fc 1" "--dropout_i 0.9 --dropout_v 0.8 --add_fc 2" "--dropout_i 0.9 --dropout_v 0.8 --add_fc 3")

configurations=(\
"--dropout_i 0.9 --dropout_v 0.8 --add_fc 1" "--dropout_i 0.9 --dropout_v 0.8 --add_fc 2" "--dropout_i 0.9 --dropout_v 0.8 --add_fc 3")


configurations=(\
"--dropout_i 0.8 --dropout_v 0.8 --add_fc 1 --use_relation_self_attention True --use_frame_self_attention True")


for extra_configurtion in "${configurations[@]}"
do
echo "$extra_configurtion"
./TestAllDomains.sh "$extra_configurtion"
done