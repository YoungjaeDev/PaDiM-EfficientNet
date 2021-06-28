# PaDiM-EfficientNet



## Requirement
* pytorch=1.8.0=py3.7_cuda11.1_cudnn8.0.5_0 (Or something similar)

## Datasets
MVTec AD datasets : Download from [MVTec website](https://www.mvtec.com/company/research/datasets/mvtec-ad/)


## Results
### Implementation results on MVTec
* Image-level anomaly detection accuracy (ROCAUC)

|MvTec|R18-Rd100|WR50-Rd550|Effi-B7-Fst-Block|Effi-B7-Md-Block|Effi-B7-Lst-Block|
|:---:|:---:|:---:|:---:|:---:|:---:|
|Carpet| 0.984| **0.999** | 0.998 | 0.998 | **0.999** |
|Grid|0.898 | 0.957| **0.992** | 0.987 | 0.988 |
|Leather|0.988 | **1.0** | **1.0** | **1.0** | 1.0 |
|Tile| 0.959| 0.974| 0.983 | 0.989 | **0.988** |
|Wood|**0.990** | 0.988| 0.987 | 0.989 | **0.990** |
|All texture classes| 0.964| 0.984| 0.992 | 0.992 | **0.993** |
|Bottle|0.996 | 0.998| **1.0** | 0.999 | 0.999 |
|Cable| 0.855| 0.922| 0.951 | 0.955 | **0.961** |
|Capsule|0.870 | 0.915| **0.924** | 0.921 | 0.915 |
|Hazelnut|0.841 |**0.933** |0.809 |0.826 |0.835 |
|Metal nut| 0.974| 0.992| **0.997** | 0.995 | 0.995 |
|Pill|0.869 | 0.944| 0.979 | **0.984** | 0.980 |
|Screw| 0.745| 0.844| 0.865 | 0.911 | **0.926** |
|Toothbrush|0.947 |**0.972** |0.967 |0.964 |0.944 |
|Transistor| 0.925| 0.978| 0.995 | **0.999** | **0.999** |
|Zipper| 0.741| 0.909| **0.930** | 0.927 | 0.922 |
|All object classes|0.876|0.941 |0.941 |**0.948** |0.947 |
|All classes| 0.905|0.955 |0.958 |**0.963** |**0.963** |

* Pixel-level anomaly detection accuracy (ROCAUC)

|MvTec|R18-Rd100|WR50-Rd550|Effi-B7-Fst-Block|Effi-B7-Md-Block|Effi-B7-Lst-Block|
|:---:|:---:|:---:|:---:|:---:|:---:|
|Carpet| 0.988| **0.990** | 0.988 | 0.985 | 0.985 |
|Grid| 0.936| 0.965| **0.979** | 0.965 | 0.963 |
|Leather|0.990 |0.989 |**0.991** |0.987 |0.987 |
|Tile|0.917 | 0.939| **0.943** | 0.929 | 0.926 |
|Wood| 0.940| 0.941| **0.944** | 0.936 | 0.934 |
|All texture classes| 0.953|0.965 |**0.969** |0.960 |0.959 |
|Bottle|0.981 | 0.982| **0.983** | 0.980 | 0.978 |
|Cable|0.949| 0.968| 0.976 | **0.978** | **0.978** |
|Capsule| 0.982| **0.986** | **0.986** | **0.986** | **0.986** |
|Hazelnut|**0.979** | **0.979** | 0.972 | 0.969 | 0.970 |
|Metal nut| 0.967|**0.971** |0.962 |0.962 |0.959 |
|Pill|0.946 |0.961 |**0.964** |0.961 |0.954 |
|Screw| 0.972| 0.983| 0.985 | **0.987** | **0.987** |
|Toothbrush|0.986 |0.987 |0.989 |**0.989** |0.988 |
|Transistor| 0.968|0.975 |0.971 |0.981 |**0.982** |
|Zipper|0.976| **0.984** | 0.976 | 0.972 | 0.971 |
|All object classes|0.971|**0.978** |0.976 |0.977 |0.975 |
|All classes| 0.965| 0.973| **0.974** | 0.971 | 0.970 |

- Test inference time(s)

| MvTec   | R18-Rd100 (original code) | WR50-Rd550 (original code) | Effi-B7 | Effi-B4 |
| ------- | ------------------------- | -------------------------- | ------- | ------- |
| Carpet  |                           | 27.519                     |         |         |
| Grid    |                           | 24.485                     |         |         |
| Leather |                           | 29.202                     |         |         |
| Tile    |                           | 26.932                     |         |         |
| Wood    |                           | 23.072                     |         |         |
| Bottle  |                           | 23.327                     |         |         |
| Cable   |                           |                            |         |         |
| Capsule |                           |                            |         |         |



 ### ROC Curve

* ResNet18

<p align="center">
    <img src="imgs/roc_curve_r18.png" width="1000"\>
</p>

* Wide_ResNet50_2

<p align="center">
    <img src="imgs/roc_curve_wr50.png" width="1000"\>
</p>
- efficientnet-b7_fst

![efficientnet-b7_fst_roc_curve](./imgs/efficientnet-b7_fst_roc_curve.png)

- efficientnet-b7_md

![efficientnet-b7_md_roc_curve](./imgs/efficientnet-b7_md_roc_curve.png)

- efficientnet-b7_lst

![efficientnet-b7_lst_roc_curve](./imgs/efficientnet-b7_lst_roc_curve.png)

### Localization examples

<p align="center">
    <img src="imgs/bottle.png" width="600"\>
</p>
<p align="center">
    <img src="imgs/cable.png" width="600"\>
</p>
<p align="center">
    <img src="imgs/capsule.png" width="600"\>
</p>
<p align="center">
    <img src="imgs/carpet.png" width="600"\>
</p>
<p align="center">
    <img src="imgs/grid.png" width="600"\>
</p>
<p align="center">
    <img src="imgs/hazelnut.png" width="600"\>
</p>
<p align="center">
    <img src="imgs/leather.png" width="600"\>
</p>
<p align="center">
    <img src="imgs/metal_nut.png" width="600"\>
</p>
<p align="center">
    <img src="imgs/pill.png" width="600"\>
</p>
<p align="center">
    <img src="imgs/screw.png" width="600"\>
</p>
<p align="center">
    <img src="imgs/tile.png" width="600"\>
</p>
<p align="center">
    <img src="imgs/toothbrush.png" width="600"\>
</p>
<p align="center">
    <img src="imgs/transistor.png" width="600"\>
</p>
<p align="center">
    <img src="imgs/wood.png" width="600"\>
</p>
<p align="center">
    <img src="imgs/zipper.png" width="600"\>
</p>

## Reference
[1]
