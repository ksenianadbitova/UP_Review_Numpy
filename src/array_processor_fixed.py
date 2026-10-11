"""Исправленная версия модуля численной обработки массивов."""
import numpy as np


class ArrayProcessor:
    """Класс для численной обработки массивов на numpy."""

    def __init__(self, data=None):
        if data is None:
            self.data = np.array([], dtype=float)
        else:
            self.data = np.asarray(data, dtype=float)

    def sum_elements(self) -> float:
        """Возвращает сумму элементов массива."""
        return float(np.sum(self.data))

    def mean_value(self) -> float:
        """Возвращает среднее значение с проверкой пустого массива."""
        if self.data.size == 0:
            raise ValueError("Невозможно вычислить среднее пустого массива")
        return float(np.mean(self.data))

    def normalize(self) -> np.ndarray:
        """Нормализация с обработкой нулевой дисперсии."""
        std = np.std(self.data)
        if std == 0:
            raise ValueError("Невозможно нормализовать массив с нулевой дисперсией")
        result: np.ndarray = (self.data - np.mean(self.data)) / std
        return result

    def find_max_index(self) -> int:
        """Индекс максимума через np.argmax."""
        if self.data.size == 0:
            raise ValueError("Массив пуст")
        return int(np.argmax(self.data))

    def cumulative_sum(self) -> np.ndarray:
        """Накопительная сумма через np.cumsum (O(n))."""
        return np.cumsum(self.data)

    def dot_product(self, other: np.ndarray) -> float:
        """Скалярное произведение с проверкой размерности."""
        other = np.asarray(other, dtype=float)
        if self.data.shape != other.shape:
            raise ValueError("Размерности массивов не совпадают")
        return float(np.dot(self.data, other))

    def moving_average(self, window: int) -> np.ndarray:
        """Скользящее среднее с валидацией окна."""
        if window <= 0:
            raise ValueError("Размер окна должен быть положительным")
        if window > self.data.size:
            raise ValueError("Размер окна превышает размер массива")
        kernel = np.ones(window) / window
        return np.convolve(self.data, kernel, mode='valid')

    def compare_floats(self, a: float, b: float, tol: float = 1e-9) -> bool:
        """Сравнение чисел с плавающей точкой с допуском."""
        return abs(a - b) < tol

    def apply_threshold(self, threshold: float) -> np.ndarray:
        """Пороговая фильтрация с обработкой NaN."""
        clean = np.nan_to_num(self.data, nan=0.0)
        result: np.ndarray = np.where(clean > threshold, clean, 0.0)
        return result

    def matrix_multiply(self, other: np.ndarray) -> np.ndarray:
        """Умножение матриц через np.matmul с проверкой размерностей."""
        other = np.asarray(other, dtype=float)
        if self.data.ndim != 2 or other.ndim != 2:
            raise ValueError("Оба массива должны быть двумерными")
        if self.data.shape[1] != other.shape[0]:
            raise ValueError("Несовместимые размерности матриц")
        return np.matmul(self.data, other)


def compute_statistics(data: list) -> dict:
    """Вычисляет статистику массива с аннотациями типов."""
    processor = ArrayProcessor(data)
    return {
        'sum': processor.sum_elements(),
        'mean': processor.mean_value(),
        'max_index': processor.find_max_index(),
        'cumsum': processor.cumulative_sum(),
    }


def safe_divide(a: float, b: float) -> float:
    """Безопасное деление с проверкой знаменателя."""
    if b == 0:
        raise ZeroDivisionError("Деление на ноль недопустимо")
    return a / b


def compute_variance(data: list) -> float:
    """Несмещенная дисперсия (ddof=1)."""
    arr = np.asarray(data, dtype=float)
    if arr.size < 2:
        raise ValueError("Для несмещенной дисперсии нужно минимум 2 элемента")
    return float(np.var(arr, ddof=1))
