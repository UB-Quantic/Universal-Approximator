import numpy as np


from .Approximant_NN import Approximant_NN
from .Approximant_NN_complex import Approximant_NN_complex
from .Approximant_Fourier import Approximant_Fourier
from .Approximant_Fourier_complex import Approximant_Fourier_complex
from .NN import NN
from .NN_complex import NN_complex
from .Fourier import Fourier
from .Fourier_complex import Fourier_complex

from .aux_functions import *

np.random.seed(0)  #Seed 0, crucial para repetir experimentos

