import numpy as np

def quadratic(x:np.array) -> np.float64:
    """Fitness functions
    Args:
        x (np.array): input variables
        
    Returns:
        np.float64: fitness value
    """
    return x[0]**2 + x[1]**2

def rastrigin(x:np.array) -> np.float64:
    """Fitness functions
    Args:
        x (np.array): input variables
        
    Returns:
        np.float64: fitness value
    """
    n = len(x)  # Número de dimensiones (tamaño del vector x)
    
    # Calculando el sumatorio
    summation = np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
    
    # Formula completa
    f = 10 * n + summation 

    return f

def ackley(x:np.array) -> np.float64:
    """Fitness functions
    Args:
        x (np.array): input variables
        
    Returns:
        np.float64: fitness value
    """
    n = len(x)

    sum1 = np.sum(x**2)

    sum2= np.sum(np.cos(2*np.pi*x))

    f = -20*np.exp(-0.2*np.sqrt(sum1/n)) - np.exp(sum2/n) + 20 + np.e

    return f