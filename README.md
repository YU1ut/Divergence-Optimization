# Divergence Optimization for Noisy UniDA
This is a PyTorch implementation of [Divergence Optimization for Noisy Universal Domain Adaptation](https://arxiv.org/abs/2104.00246). 

## Requirements
- Python 3.8
- PyTorch 1.6.0
- torchvision 0.7.0
- matplotlib
- numpy
- scikit-learn

## Preparation
Downlaod following data:

[Office](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/)

[OfficeHome](http://hemanthdv.org/OfficeHome-Dataset/) 

[VisDA](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification)

and put them in data directory as follows:
```
Divergence-Optimization
│   README.md
│   train.py
│   run.py
│   ...
│   
└───data
    └───amazon
    |   └───images
    └───dslr
    └───webcam
    └───Art
    |   └───Alarm_Clock
    └───Clipart
    └───Product
    └───Real
    └───visda
        └───train
        └───validation
        
```

## Usage
Train the network with Office dataset under Noisy UniDA setting having pairflip noise (noise rate = 0.2):
 
```
python run.py --gpu 0 --dataset office --noise-type pairflip --percent 0.2
```

The trained model and output will be saved at `result/pairflip_0.2/configs/office-train-config_opda`.

**For more details and parameters, please refer to --help option.**

## Reference codes
- https://github.com/VisionLearningGroup/DANCE

## References
- [1]: Qing Yu, Atsushi Hashimoto and Yoshitaka Ushiku. "Divergence Optimization for Noisy Universal Domain Adaptation", in CVPR, 2021.