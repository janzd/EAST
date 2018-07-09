
# EAST: An Efficient and Accurate Scene Text Detector

This is a Keras implementation of EAST based on a Tensorflow implementation made by [argman](https://github.com/argman/EAST).

The original paper by Zhou et al. is available on [arxiv](https://arxiv.org/abs/1704.03155).

+ Only RBOX geometry is implemented
+ Differences from the original paper
    + Uses ResNet-50 instead of PVANet
    + Uses dice loss function instead of balanced binary cross-entropy
    + Uses AdamW optimizer instead of the original Adam

The implementation of AdamW optimizer is borrowed from [this repository](https://github.com/shaoanlu/AdamW-and-SGDW).

The code should run under both Python 2 and Python 3.

### Requirements

Keras 2.0 or higher, and TensorFlow 1.0 or higher should be enough.

### Data

You can use your own data, but the annotation files need to conform the ICDAR 2015 convention.

ICDAR 2015 dataset can be downloaded from this [site](http://rrc.cvc.uab.es/?ch=4&com=introduction). You need the data from Task 4.1 Text Localization.\
You can also download the [MLT dataset](http://rrc.cvc.uab.es/?ch=8&com=introduction), which uses the same annotation style as ICDAR 2015, there.

### Training

You need to put all of your training images and their corresponding annotation files in one directory. The annotation files have to be named `gt_IMAGENAME.txt`.\
You also need a directory for validation data, which requires the same structure as the directory with training images.

```
python train.py --gpu_list=0,1 --input_size=512 --batch_size=12 --num_workers=6 --training_data_path=../data/ICDAR2015/train_data/ --validation_data_path=../data/MLT/val_data_latin/ --checkpoint_path=tmp/icdar2015_east_resnet50/
```

### Test

The images you want to classify have to be in one directory, whose path you have to pass as an argument.

```
python eval.py --gpu_list=0 --test_data_path=../data/ICDAR2015/test/ --model_path=tmp/icdar2015_east_resnet50/ --output_dir=tmp/icdar2015_east_resnet50/eval/
```

README: TO BE FINISHED SOON
