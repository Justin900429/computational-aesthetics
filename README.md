# README

## Introduction
This repo is to extract the computational aesthetic features introduced from the paper - ["The Pictures we Like are our Image: Continuous Mapping of Favorite Pictures into Self-Assessed and Attributed Personality Traits"](https://ieeexplore.ieee.org/document/7378902). Some of the features had been removed and readapted. See the table below.

## Feature Synopis
| Category            | Name                                                                                                                      | dimension                                       |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| Color               | HSV statics <br/> Emotion-based <br/> Color diversity <br/> Color name                                                    | 5 <br/> 3 <br/> 1 <br/> 11                      |
| Composition         | Edge pixels <br/> Level of detial <br/> Average region size <br/> Low depth of ﬁeld (DOF) <br/> Rule of thirds Image size | 1 <br/> 1 <br/> 1 <br/> 3 <br/> 2 <br/> 1 <br/> |
| Textural Properties | Gray distribution entropy <br/> Wavelet based textures <br/> Tamura <br/> GLCM - features                                 | 1 <br/> 12 <br/> 3 <br/> 3                      |

> The description column is omitted. To see what each feature does, please refer to the Paper. Additionally, there are some changes being made.
> 1. The **Faces features** was removed.
> 2. The realization of **color diversity** is different from the paper. 
> 3. The **GIST descriptors** was removed.
> 4. The **GLCM - features** used only gray image and left only 4 features.

## How to use ?

### Install requirements
```
$ pip install git+https://github.com/Justin900429/computational-aesthetics
```

### Import file
```python
from CA import CA
...
# Create objects
img_path = "..."
ca = CA(img_path)
res = ca.compute_ca()
...
# update image path
new_path = "..."
ca.update(new_path)
new_res = ca.compute_ca()
...
```

See [example](https://github.com/Justin900429/computational-aesthetics/blob/main/example.py) for more details

## Citation
> Apology for not including all the citations. Below section only lists the paper mentioned in the introduction section. All the reference are included in the below paper.

```bibtex
@ARTICLE{
  7378902,
  author={Segalin, Crisitina and Perina, Alessandro and Cristani, Marco and Vinciarelli, Alessandro},
  journal={IEEE Transactions on Affective Computing},
  title={The Pictures We Like Are Our Image: Continuous Mapping of Favorite Pictures into Self-Assessed and Attributed Personality Traits},
  year={2017},
  volume={8},
  number={2},
  pages={268-285},
  doi={10.1109/TAFFC.2016.2516994}
}
```
