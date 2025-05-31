## **CSI-DMT:Multi-Focus Image Fusion via Cross-Task Semantic Interaction and Dual-Attention Mixing Transformer**

### Submission to The Visual Computer Journal

This repository contains the implementation of CSI-DMT,a network that uses cross-task semantic interaction and dual-attention mixing transformer for multi-focus image fusion. To evaluate the effectiveness of our method, we compared it with other state-of-the-art methods on four multi-focus image datasets,achieving state-of-the-art results in both subjective visual effects and objective measurement outcomes, while also demonstrating significant advantages in lightweight design and time efficiency. The model weights will be uploaded after the paper is accepted.

### Document

`CSI-DMT/`
`├── .idea/`
`├── _pycache_/`
`├── Datasets/`
`│   ├── Eval/`
`│          └── Lytro`
`│                 ├── sourceA/`
`│                        ├── lytro-01-A.jpg`
`│                        ├── lytro-02-A.jpg`
`│                        ├── ...
│                        ├── lytro-20-A.jpg
│                 └── sourceB/`
`│                        ├── lytro-01-B.jpg`
`│                        ├── lytro-02-B.jpg`
`│                        ├── ...    
│                        ├── lytro-20-B.jpg`
`│   └── Train&Valid/`
`│          └── DUTS_MFF/`
`│                 ├── train/`
`│                        ├── decisionmap/`
`│                        ├── groundtruth/`
`│                        ├── sourceA/`
`│                        └── sourceB/`
`│                 └── validate`
`│                        ├── decisionmap/`
`│                        ├── groundtruth/`
`│                        ├── sourceA/`
`│                        └── sourceB/`
`├── Loss_funcs/`
`├── Nets/`
`│   ├── _init_.py`
`│   ├── CSI_DMT.py`
`│   ├── DACF.py`
`│   └── DAMT.py`
`├── RunTimeData/`
`│   └── Model weights`
`│          └── best_network.pth`
`├── Utilities/`
`├── Fusion.py`
`├── image_fusion_gui.py`
`├── README.md`
`└── Training.py`

| File name           | Explanation                                                  |
| ------------------- | ------------------------------------------------------------ |
| Datasets            | Training dataset and the testing dataset(There are no restrictions on image naming, they just need to correspond one-to-one in the folder) |
| Loss_funcs          | Loss function                                                |
| Nets                | Network structure                                            |
| Results             | Test output                                                  |
| RunTimeData         | Trained model weights                                        |
| Utilities           | Utility function                                             |
| Fusion.py           | For testing                                                  |
| Training.py         | For training                                                 |
| image_fusion_gui.py | Graphical user interface                                     |

### Preparation

#### Dependencies

- python=3.9.21
- pytorch=2.5.1
- torchvision =0.20.1
- numpy=2.0.1
- opencv=4.10.0.84
- pillow =11.0.0
- scipy = 1.13.1 

#### Data Preparation

[DUTS](https://paperswithcode.com/dataset/duts)(training dataset)

[Lytro](https://mansournejati.ece.iut.ac.ir/content/lytro-multi-focus-dataset)(testing dataset)

[Grayscale](https://github.com/yuliu316316/MFIF/tree/master/sourceimages/grayscale)(testing dataset)

[MFFW](https://github.com/lmn-ning/ImageFusion/tree/main/FusionDiff/Dataset/Multi-Focus-Images/valid)(testing dataset)

[RealMFF](https://github.com/Zancelot/Real-MFF)(testing dataset)

### Training

For training, please run:

`python Training.py`

### Testing

For testing, please run:

`python Fusion.py`

### Graphical User Interface

For launching the graphical user interface,please run:

`python image_fusion_gui.py`

### Results

The output results will be stored in ./Results
