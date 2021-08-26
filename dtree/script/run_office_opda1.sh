#!/bin/sh  webcam  dslr  amazon
#export LD_LIBR RY_PATH=$LD_LIBRARY_PATH:/home1/suwan/anaconda3/lib/python3.8/site-packages/scipy/fft/_pocketfft
python train_dance.py --config $2 --source ./txt/source_amazon_opda.txt --target ./txt/target_dslr_opda.txt --gpu $1