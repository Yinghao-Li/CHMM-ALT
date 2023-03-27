#!/bin/bash

# Author: Yinghao Li @ Georgia Tech ECE
# All rights reserved

# Quit if there are any errors
set -e

for PARTITION in 'train' 'test'
do
python data_build.py --partition $PARTITION
done

python update_dataset.py
