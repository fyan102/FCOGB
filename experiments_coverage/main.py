from experiments_coverage.evaluate_dataset_coverage import evaluate_datasets
from experiments_coverage.evaluate_friedman_coverage import evaluate_friedman
from experiments_coverage.evaluate_loaded_coverage import eveluate_loaded
from experiments_coverage.evaluate_poisson_coverage import evaluate_poisson

if __name__=='__main__':
    evaluate_datasets()
    eveluate_loaded()
    evaluate_friedman()
    evaluate_poisson()