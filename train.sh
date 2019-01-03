#!/bin/sh

python train.py --gpu_list=1 --input_size=512 --batch_size=12 --nb_workers=6 --training_data_path=../../data/ICDAR2015/train_data/ --validation_data_path=../../data/MLT/val_data_latin/ --checkpoint_path=tmp/icdar2015_east_resnet50/
