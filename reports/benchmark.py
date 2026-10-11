import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import timeit
import numpy as np
from src.array_processor import ArrayProcessor

data = list(range(10000))
arr = np.array(data)
proc = ArrayProcessor(data)

t1 = timeit.timeit(lambda: proc.sum_elements(), number=100)
print(f"sum_elements (цикл): {t1:.4f} сек")

t2 = timeit.timeit(lambda: np.sum(arr), number=100)
print(f"np.sum (векторизация): {t2:.4f} сек")

t3 = timeit.timeit(lambda: proc.cumulative_sum(), number=10)
print(f"cumulative_sum (O(n^2)): {t3:.4f} сек")

t4 = timeit.timeit(lambda: np.cumsum(arr), number=10)
print(f"np.cumsum (векторизация): {t4:.4f} сек")