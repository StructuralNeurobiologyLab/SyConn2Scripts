import pandas
from scipy.stats import ks_2samp, ranksums
import numpy as np


if __name__ == '__main__':
    df = pandas.read_excel('/home/pschuber/pCloudDrive/Doktorarbeit/materials/SyConn v2 paper/figures/1/'
                         'celltype_performance/example_result_redundancy50_points50k_ctx20um/redun50_certainty.xls')
    cert = df['certainty'].to_numpy()
    correct = df['quantity'].to_numpy()
    res = ranksums(cert[correct == 'correct'], cert[correct == 'incorrect'])
    print('#correct:', np.sum(correct == 'correct'), '#incorrect:', np.sum(correct == 'incorrect'))
    print(res)
