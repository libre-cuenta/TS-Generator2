import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.interpolate import CubicSpline

def TRENDGenerator(f):
    def TRENDpsd(time, dtype=[float, int], **kwargs):
        if len(time) < 1 :
            raise TypeError("Необходимо задать кременной промежуток для генерации")
        
        Y = f(time, **kwargs)
        
        if (dtype == int):
            Y = list(map(int, Y))
        
        return Y
    return TRENDpsd

@TRENDGenerator
def polinom_trend(time, a=(1,)):
    if ((type(a) is int) or (type(a) is float)):
        return [a] * len(time)
    
    Y = list()
    for x in time:
        tDegrees = [(x**i) for i in range(len(a))]
        Y.append(np.dot(a , tDegrees))
    
    return Y

@TRENDGenerator
def exp_trend(time, a=1, b=1):
    Y = list()
    for x in time:
        Y.append(a*math.exp(b*x))

    return Y

@TRENDGenerator
def exp2_trend(time, core, a=1, b=1):
    Y = list()
    for x in time:
        Y.append(a * core**(b*x))

    return Y

@TRENDGenerator
def log_trend(time, a=1, b=1):
    Y = list()
    time_positive = filter(lambda x: x > 0, time)

    for x in time_positive:
        Y.append(a+b*np.log(x))

    return Y

@TRENDGenerator
def extend_trend(time, a=1, b=1):
    Y = list()
    for x in time:
        Y.append(a*(x**b))

    return Y


def spline_trend(x_points, y_points, num_points=100):
    if len(x_points) != len(y_points):
        raise ValueError("Количество x и y координат должно совпадать")
    
    if len(x_points) < 3:
        raise ValueError("Для построения кубического сплайна необходимо минимум 3 точки")
    
    cs = CubicSpline(x_points, y_points)
    

    x_new = np.linspace(min(x_points), max(x_points), num_points)
    y_new = cs(x_new)
    
    return x_new, y_new



