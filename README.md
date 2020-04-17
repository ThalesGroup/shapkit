# Shapkit
> Summary description here.


This file will become your README and also the index of your documentation.

## Install

`pip install your_project_name`

## How to use

Fill me in please! Don't forget code examples:

### Regression example

We use a simulated dataset from the book _Elements of Statistical Learning_ ([hastie,2009], the Radial example). $X_1, \dots , X_{d}$ are standard independent Gaussian. The model is determined by:

{% raw %}
$$ Y = \prod_{j=1}^{d} \rho(X_j), $$
{% endraw %}

where $\rho\text{: } t \rightarrow \sqrt{(0.5 \pi)} \exp(- t^2 /2)$. The regression function $f_{regr}$ is deterministic and simply defined by $f_r\text{: } \textbf{x} \rightarrow \prod_{j=1}^{d} \phi(x_j)$. For a reference $\mathbf{r^*}$ and a target $\mathbf{x^*}$, we define the reward function $v_r^{\mathbf{r^*}, \mathbf{x^*}}$ such as for each coalition $S$, $v_r^{\mathbf{r^*}, \mathbf{x^*}}(S) = f_{regr}(\mathbf{z}(\mathbf{x^*}, \mathbf{r^*}, S)) - f_{regr}(\mathbf{r^*}).$

 [hastie,2009] _The Elements of Statistical Learning: Data Mining, Inference, and Prediction, Second Edition_. Hastie, Trevor and Tibshirani, Robert and Friedman, Jerome. Springer Series in Statistics, 2009.
	

```python
def generate_sample(dim, n_samples, rho=0):
    """
    Generate a dataset of independent Gaussian features
    """
    mu = np.zeros(dim)
    sigma = np.ones((dim, dim)) * rho
    np.fill_diagonal(sigma, [1] * dim)
    # Simulation
    X = np.random.multivariate_normal(mean=mu, cov=sigma, size=n_samples)
    df_X = pd.DataFrame(X, columns=['x'+str(i) for i in range(1, dim+1)])
    return df_X
```

```python
d, n_samples = 5, 100
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


```python
idx_r, idx_x = np.random.choice(np.arange(len(X)), size=2, replace=False)
r = X.iloc[idx_r,:]
x = X.iloc[idx_x,:]
```

```python
from shapkit_nbdev.shapley_values import ShapleyValues
from shapkit_nbdev.monte_carlo_shapley import MonteCarloShapley
from shapkit_nbdev.sgd_shapley import SGDshapley
```

#### Shapley Values

```python
true_shap = ShapleyValues(x=x, fc=fc, r=r)
```

    100%|██████████| 5/5 [00:00<00:00, 483.33it/s]


```python
true_shap
```




    x1    0.051504
    x2   -0.059734
    x3    0.165201
    x4   -0.167416
    x5    0.149816
    dtype: float64



#### Monte Carlo

```python
mc_shap = MonteCarloShapley(x=x, fc=fc, r=r, n_iter=100, callback=None)
```

    100%|██████████| 100/100 [00:00<00:00, 4785.18it/s]


```python
mc_shap
```




    x1    0.049357
    x2   -0.054322
    x3    0.162383
    x4   -0.169489
    x5    0.151442
    dtype: float64



#### Projected Stochastic Gradient Shapley 

```python
sgd_est = SGDshapley(d, C=y.max())
sgd_shap = sgd_est.sgd(x=x, fc=fc, r=r, n_iter=1000, step=.1, step_type="sqrt")
```

    100%|██████████| 1000/1000 [00:00<00:00, 8021.88it/s]


```python
sgd_shap
```




    x1    0.050282
    x2   -0.048745
    x3    0.149134
    x4   -0.147864
    x5    0.136565
    dtype: float64


