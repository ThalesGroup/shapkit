# Project name here
> Summary description here.


This file will become your README and also the index of your documentation.

## Install

`pip install your_project_name`

## How to use

Fill me in please! Don't forget code examples:

### Toy dataset

```python
def generate_sample(dim, n_samples, rho=0):
    """
    Gaussian Features
    ex: dim = 3
    mu=[0, 0, 0]
    sigma = [1 rho rho]
            [rho 1 rho]
            [rho rho 1]
    p(x) ~ N(mu, sigma)
    """
    # Law parameters
    mu = np.zeros(dim)
    sigma = np.ones((dim, dim)) * rho
    np.fill_diagonal(sigma, [1] * dim)
    # Simulation
    X = np.random.multivariate_normal(mean=mu, cov=sigma, size=n_samples)
    df_X = pd.DataFrame(X, columns=['x'+str(i) for i in range(1, dim+1)])
    return df_X
```

```python
d = 5
n_samples = 100
X = generate_sample(d, n_samples)
y = np.zeros(len(X))
for i in range(len(X)):
    phi_x = np.sqrt(.5 * np.pi) * np.exp(-0.5 * X.values[i] ** 2)
    y[i] = np.prod(phi_x)
```

```python
n = 2**d - 2
def fc(x):
    phi_x = np.sqrt(.5 * np.pi) * np.exp(-0.5 * x ** 2)
    return np.prod(phi_x)
print("dimension = {0} ; nb of coalitions = {1}".format(str(d), str(n)))
```

    dimension = 5 ; nb of coalitions = 30


### Illustration

```python
# set the game
idx_r, idx_x = np.random.choice(np.arange(len(X)), size=2, replace=False)
r = X.iloc[idx_r,:]
x = X.iloc[idx_x,:]
```

```python
true_shap = ShapleyValues(x=x, fc=fc, r=r)
```

    100%|██████████| 5/5 [00:00<00:00, 537.91it/s]


```python
true_shap
```




    x1   -0.212646
    x2   -0.210187
    x3   -0.224681
    x4    0.569841
    x5   -0.193766
    dtype: float64


