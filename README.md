## Orthogonal Gradient Boosting for Interpretable Additive Rule Ensembles

This repository contains code, datasets, and supplementary information for the paper.

### Setup
To replicate the experiments, you have to have Python of version at 3.9.11 installed on your machine. In addition you have to install dependencies via:

```
pip3 install -r requirements.txt
```

You need to add the ```FCOGB``` directory to the ```PYTHONPATH``` environment variable.

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


