#!/bin/bash




extra_parameters=$1
# Test All Modalities
./TestAllModalities.sh D1 D2 "$extra_parameters"
./TestAllModalities.sh D1 D3 "$extra_parameters"

./TestAllModalities.sh D2 D1 "$extra_parameters"
./TestAllModalities.sh D2 D3 "$extra_parameters"

./TestAllModalities.sh D3 D1 "$extra_parameters"
./TestAllModalities.sh D3 D2 "$extra_parameters"

#./TestAllModalities.sh D1 D1 "$extra_parameters"
#./TestAllModalities.sh D2 D2 "$extra_parameters"
#./TestAllModalities.sh D3 D3 "$extra_parameters"