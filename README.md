# TME

This repository contains a simple examples of the tensor mixture of experts (TME) approach.

## Dependencies

The code was tested with Python 3.6 and uses the following libraries:
* numpy
* scipy
* matplotlib
* pillow

## Examples
    - demo_mme
    This example generates data according to matrix mixture of experts model based on matrix coefficients.
    The recovery of these coefficients is tested with Ridge regression, matrix Ridge regression, 
    a mixture of experts model and a matrix mixture of experts model.

    - demo_tme
    This example generates data according to tensor mixture of experts model based on tensor coefficients.
    The recovery of these coefficients is tested with Ridge regression, tensor Ridge regression, 
    a mixture of experts model and a tensor mixture of experts model.
    
## Reference
If you found this useful, we would be grateful if you cite the following [reference](http://njaquier.ch/files/TensorMixtExp_arXiv.pdf):

[1] N. Jaquier, R. Haschke and S. Calinon (2019). *Tensor-variate Mixture of Experts.* ArXiv preprint 1902.11104.
```
@article{Jaquier19:TME,
	author = {Jaquier, N. and Haschke, R. and Calinon, S.},
	title = {Tensor-variate Mixture of Experts},
	booktitle = {arXiv preprint 1902.11104},
	year = {2019},
	pages = {},
}
```