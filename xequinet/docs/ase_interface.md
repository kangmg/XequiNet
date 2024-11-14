## Types of XPaiNN Model

XPaiNN models can be JIT compiled in two ways:

```shell
xeqjit -c model.pt -o ./geo_model.jit # for the `geometry` model

xeqjit -c model.pt -o ./md_model.jit --md # for the `md` model
```

Each model type has a different input structure, so specify the `model_type` as either `geometry` or `md` when loading the model in ASE.

## ASE Calculator for XPaiNN

The XPaiNN model can be loaded from the `ckpt_file` path or by passing the model directly as the `model` parameter, as follows.

```python
from xequinet.interface import XeqCalculator
import torch

# from `ckpt_file`
calc = XeqCalculator(ckpt_file='md_model.jit', model_type='md')

# pass the model object
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.jit.load("geo.jit", map_location=device)
calc = XeqCalculator(model=model, model_type='geometry')
```


## Using SumCalculator for $\Delta$-Learning Models

The $\Delta$-learning model must be used with the TBLite Calculator. Here's an example:

```python
from ase.calculators.mixing import SumCalculator
from xequinet.interface import XeqCalculator
from tblite.ase import TBLite


calc = SumCalculator([
    XeqCalculator(ckpt_file='d_md.jit', model_type='md'), # delta learning model
    TBLite(verbosity=0, method='GFN2-xTB'), # xtb
])
```