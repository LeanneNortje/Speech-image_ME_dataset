Before you get started download the following speech and image datasets at the links provided. Make sure to extract the datasets to have the correct structure or change the code accordingly.

# Image datasets:
* ## Caltech-101
  
  [Download](https://data.caltech.edu/records/mzrjq-6wc02)
  
  Structure:

```bash
├── MACOSX
│   ├── ._caltech-101
│   ├── caltech-101
│   │   ├── ._101_ObjectCategories.tar.gz
│   │   ├── ._Annotations.tar
│   │   ├── ._show_annotation.m
├── caltech-101
│   ├── 101_ObjectCategories.tar.gz
│   ├── Annotations.tar
│   ├── show_annotation.m
```

* ## ImageNet
  
  Structure:

```bash
├── ILSVRC
│   ├── Annotations
│   ├── Data
│   │   ├── test 
│   │   │   ├── ILSVRC2012_test_00033334.JPEG
│   │   │   ├── ILSVRC2012_test_00050001.JPEG
│   │   │   ├── ...
│   │   │   ├── ILSVRC2012_test_00066668.JPEG
│   │   │   ├── ILSVRC2012_test_00083335.JPEG
│   │   ├── train
│   │   │   ├── n01917289_968.JPEG
│   │   │   ├── n02104029_956.JPEG
│   │   │   ├── ...
│   │   │   ├── n02342885_1950.JPEG
│   │   │   ├── n02834397_34349.JPEG 
│   │   ├── val 
│   │   │   ├── ILSVRC2012_val_00007143.JPEG
│   │   │   ├── ILSVRC2012_val_00014286.JPEG
│   │   │   ├── ...
│   │   │   ├── ILSVRC2012_val_00021429.JPEG
│   │   │   ├── ILSVRC2012_val_00028572.JPEG
│   ├── LOC_synset_mapping.txt
```
  
To download the content for the ```train```, ```val```, ```test``` and ```Annotation``` folder, go [here]([https://data.caltech.edu/records/mzrjq-6wc02](https://image-net.org/download-images.php)).
  
For the training samples, download the ```Training images (Task 1 & 2)``` zip file under the 2012 Images of ```ImageNet Large-scale Visual Recognition Challenge (ILSVRC)```.
For the validation images, download the ```Validation images (all tasks)``` zip file under the 2012 Images of ```ImageNet Large-scale Visual Recognition Challenge (ILSVRC)```.
For the testing images, download the ```Test images (all tasks).``` zip file under the 2012 Images of ```ImageNet Large-scale Visual Recognition Challenge (ILSVRC)```.
The Annotation files can be downloaded under ```Training bounding box annotations (Task 1 & 2 only) ``` under the bounding boxes of the 2012 ```ImageNet Large-scale Visual Recognition Challenge (ILSVRC)```.
Lastly, download the ```LOC_synset_mapping.txt``` file [here](https://github.com/formigone/tf-imagenet/blob/master/LOC_synset_mapping.txt).

* ## MSCOCO
