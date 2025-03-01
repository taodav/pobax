from scipy.stats import sem, t
from scipy.signal import savgol_filter

colors = {
    'pink': '#ff96b6',
    'red': '#df5b5d',
    'orange': '#DD8453',
    'yellow': '#f8de7c',
    'green': '#3FC57F',
    'dark green': '#27ae60',
    'cyan': '#48dbe5',
    'blue': '#3180df',
    'purple': '#9d79cf',
    'brown': '#886a2c',
    'white': '#ffffff',
    'light gray': '#d5d5d5',
    'dark gray': '#666666',
    'black': '#000000'
}

def mean_confidence_interval(data, confidence=0.95, axis=-1):
    n = data.shape[axis]
    m, se = data.mean(axis=axis), sem(data, axis=axis)
    h = se * t.ppf((1 + confidence) / 2., n-1)
    return m, h

def smoothen(data, window: int = 30, polynomial_deg: int = 3):
    return savgol_filter(data, window, polynomial_deg)
