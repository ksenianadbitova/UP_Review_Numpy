import numpy as np
import pytest
from src.array_processor import ArrayProcessor, safe_divide, compute_variance


def test_empty_array_mean():
    """Дефект 1: mean_value не обрабатывает пустой массив."""
    proc = ArrayProcessor([])
    with pytest.raises((ZeroDivisionError, ValueError)):
        proc.mean_value()


def test_normalize_zero_variance():
    """Дефект 2: normalize не обрабатывает нулевую дисперсию."""
    proc = ArrayProcessor([5, 5, 5, 5])
    with pytest.raises((ZeroDivisionError, ValueError)):
        proc.normalize()


def test_compare_floats_incorrect():
    """Дефект 3: сравнение чисел с плавающей точкой через ==."""
    proc = ArrayProcessor([0.1, 0.2, 0.3])
    assert proc.compare_floats(0.1 + 0.2, 0.3) is True


def test_cumulative_sum_complexity():
    """Дефект 4: cumulative_sum имеет сложность O(n^2)."""
    import time
    data = list(range(5000))
    proc = ArrayProcessor(data)
    start = time.time()
    proc.cumulative_sum()
    elapsed = time.time() - start
    assert elapsed < 0.1, f"cumulative_sum слишком медленный: {elapsed:.4f} сек"


def test_safe_divide_by_zero():
    """Дефект 5: safe_divide не обрабатывает деление на ноль."""
    with pytest.raises(ZeroDivisionError):
        safe_divide(10, 0)


def test_compute_variance_bias():
    """Дефект 6: compute_variance использует смещенную оценку."""
    data = [1, 2, 3, 4, 5]
    biased = compute_variance(data)
    unbiased = np.var(data, ddof=1)
    assert abs(biased - unbiased) < 1e-10, "Дисперсия вычислена со смещением"


def test_moving_average_invalid_window():
    """Дефект 7: moving_average не обрабатывает window <= 0."""
    proc = ArrayProcessor([1, 2, 3, 4, 5])
    with pytest.raises(ValueError):
        proc.moving_average(0)


def test_dot_product_mismatch():
    """Дефект 8: dot_product возвращает None вместо исключения."""
    proc = ArrayProcessor([1, 2, 3])
    with pytest.raises(ValueError):
        proc.dot_product([1, 2])


def test_apply_threshold_with_nan():
    """Дефект 9: apply_threshold не учитывает NaN."""
    proc = ArrayProcessor([1.0, np.nan, 3.0])
    result = proc.apply_threshold(2.0)
    assert not np.any(np.isnan(result)), "NaN не обработан"


def test_matrix_multiply_dimension_mismatch():
    """Дефект 10: matrix_multiply не проверяет размерности."""
    proc = ArrayProcessor(np.eye(3))
    with pytest.raises(ValueError):
        proc.matrix_multiply(np.eye(2))
