## Orthogonal Gradient Boosting for Interpretable Additive Rule Ensembles

This repository contains code, datasets, and supplementary information for the paper.

To run the experiments of comparing risks and scores of different algorithms: 

```
    cd experiments
    python combined.py
```

To run the experiments of comparing computational times of different algorithms: 

```
    cd experiments
    python time_combined.py
```

To compare the coverages: 

```
    cd experiments_coverage
    python main.py
```

The experiments for SIRUS are run in R language, the location of the file is: ```./sirus_experiments/sirus_experiments_datasets.R```.

The analysis are in the ```analysis``` folder.

If there is an error saying "no module named ...", add the ```FCOGB``` directory to the ```PYTHONPATH``` environment variable.
