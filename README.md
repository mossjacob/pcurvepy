# pcurvepy
### Principal Curves 

This is an implementation of the Principal Curves (Hastie '89) algorithm in Python.

It is a fork of the `zsteve/pcurvepy` package where 
the projection indices are selected according to our 
translation of the R/C++ `princurve` package.

Installation:
```
pip install git+https://github.com/mossjacob/pcurvepy
```

Example:

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pcurve

data = pd.read_csv('test_data.csv')
x = data.loc[:, ('X1', 'X2')].to_numpy()

# transform data to have zero mean
x = x - np.mean(x, 0)
index = np.arange(0, len(x))

curve = pcurve.PrincipalCurve(k=5)
curve.fit(x)

plt.scatter(x[:, 0], x[:, 1], alpha=0.25, c=index)
plt.plot(curve.points[:, 0], curve.points[:, 1], c='k')

# get interpolation indices
pseudotime_interp, point_interp, order = curve.unpack_params()


```

![example](example.png)
