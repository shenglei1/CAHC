"""使用现有实验记录生成中文图，不修改源数据。"""
import matplotlib
matplotlib.use('Agg')
from generate_main_results_zh import main as main_results
from generate_additional_figures_zh import main as additional
if __name__ == '__main__':
    main_results()
    additional()
