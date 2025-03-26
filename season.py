import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.interpolate import CubicSpline

def SEASONGenerator(f):
    def SEASONpsd(time, dtype=[float, int], *args):
        if len(time) < 1:
            raise TypeError("Необходимо задать кременной промежуток для генерации")
        
        Y = f(time, *args)
        
        if (dtype == int):
            Y = list(map(int, Y))
        
        return Y
    return SEASONpsd

@SEASONGenerator
def season_1(time, a, b, c, d):
    Y = list()
    for x in time:
        Y.append( a * (np.abs(np.sin(b*x+c)))**(d) * np.sign(np.sin(b*x+c)) )
        
    return Y

@SEASONGenerator
def season_2(time, a0, a, b, alpha, delta=1):
    # Проверка входных данных
    if len(a) != len(b):
        raise ValueError("Количество x и y координат должно совпадать")
    
    Y = list()
    for x in time:
        season = 0
        for i in range(len(a)):
            season += a[i]*np.cos(alpha*i*(x**delta)) + b[i]*np.sin(alpha*i*(x**delta))

        Y.append(a0 + season)

    return Y
