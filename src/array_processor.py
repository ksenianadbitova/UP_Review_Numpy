import numpy as np


class ArrayProcessor:
    """Класс для численной обработки массивов."""

    def __init__(self, data=None):
        if data is None:
            self.data = np.array([])
        else:
            self.data = np.array(data)

    def sum_elements(self):
        """Суммирование элементов массива через цикл."""
        total = 0.0
        for i in range(len(self.data)):
            total = total + self.data[i]
        return total

    def mean_value(self):
        """Среднее значение с делением на длину без проверки."""
        total = self.sum_elements()
        return total / len(self.data)

    def normalize(self):
        """Нормализация массива без обработки нулевой дисперсии."""
        result = []
        mean = self.mean_value()
        for i in range(len(self.data)):
            result.append((self.data[i] - mean) / np.std(self.data))
        return np.array(result)

    def find_max_index(self):
        """Поиск индекса максимума через ручной цикл."""
        max_val = self.data[0]
        max_idx = 0
        for i in range(1, len(self.data)):
            if self.data[i] > max_val:
                max_val = self.data[i]
                max_idx = i
        return max_idx

    def cumulative_sum(self):
        """Накопительная сумма через вложенные циклы O(n^2)."""
        result = np.zeros(len(self.data))
        for i in range(len(self.data)):
            s = 0
            for j in range(i + 1):
                s = s + self.data[j]
            result[i] = s
        return result

    def dot_product(self, other):
        """Скалярное произведение через цикл."""
        if len(self.data) != len(other):
            return None
        result = 0
        for i in range(len(self.data)):
            result = result + self.data[i] * other[i]
        return result

    def moving_average(self, window):
        """Скользящее среднее с необработанным граничным случаем."""
        result = []
        for i in range(len(self.data)):
            if i < window:
                result.append(np.mean(self.data[0:i + 1]))
            else:
                result.append(np.mean(self.data[i - window + 1:i + 1]))
        return np.array(result)

    def compare_floats(self, a, b):
        """Сравнение чисел с плавающей точкой через ==."""
        return a == b

    def apply_threshold(self, threshold):
        """Пороговая фильтрация без учета NaN."""
        result = []
        for x in self.data:
            if x > threshold:
                result.append(x)
            else:
                result.append(0.0)
        return np.array(result)

    def matrix_multiply(self, other):
        """Умножение матриц через тройной цикл."""
        n = len(self.data)
        m = len(other[0])
        k = len(other)
        result = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                for p in range(k):
                    result[i][j] = result[i][j] + self.data[i][p] * other[p][j]
        return result


def compute_statistics(data):
    """Функция вычисления статистики без аннотаций типов."""
    processor = ArrayProcessor(data)
    stats = {}
    stats['sum'] = processor.sum_elements()
    stats['mean'] = processor.mean_value()
    stats['max_index'] = processor.find_max_index()
    stats['cumsum'] = processor.cumulative_sum()
    return stats


def safe_divide(a, b):
    """Деление без проверки на ноль."""
    return a / b


def compute_variance(data):
    """Вычисление дисперсии с ошибкой смещения."""
    n = len(data)
    mean = sum(data) / n
    total = 0
    for x in data:
        total = total + (x - mean) ** 2
    return total / n
