# Datasets

Most notebooks in this series are **data-free** — the differential equation *is*
the training signal, and where reference solutions are needed they are computed
analytically inside the notebook (e.g. the Taylor–Green vortex in notebook 10).

You only need external data to reproduce the *original* benchmark results from
Raissi et al. (2019):

| File | Used by | Source |
|---|---|---|
| `burgers_shock.mat` | notebook 04 (optional accuracy check) | https://github.com/maziarraissi/PINNs |
| `cylinder_nektar_wake.mat` | notebook 10 (original cylinder-wake variant) | https://github.com/maziarraissi/PINNs |

To fetch them:

```bash
git clone https://github.com/maziarraissi/PINNs.git
cp PINNs/main/Data/*.mat datasets/
```

Load `.mat` files with `scipy.io.loadmat`. The notebooks explain exactly where to
substitute them.
