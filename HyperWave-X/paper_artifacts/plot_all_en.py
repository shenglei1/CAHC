"""Reproduce English figures using existing experimental evidence."""
import matplotlib
matplotlib.use('Agg')
from generate_main_results import main as main_results
from generate_additional_figures import main as additional
if __name__ == '__main__':
    main_results()
    additional()
