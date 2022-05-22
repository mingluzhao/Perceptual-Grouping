import os
import sys
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))
sys.path.append(os.path.join(dirName, '..'))
import pygame
import random
import pandas as pd
import matplotlib.pyplot as plt
from pygame.locals import *
from sys import exit
from src.functionWarGame import *
from src.simulateCPEP import *


def stepFunction(x):
    if x>0:
        return 1
    else:
        return 0


class SimulateByWarField:

    def __init__(self, length):
        self.formerWarField = [0 for i in range(length)]
        self.deltaWarField = [0 for i in range(length)]
        self.length = length

    def __call__(self, warField, remainingSoldiersA, remainingSoldiersB, soldiersA,
                                                   soldiersB):
        for i in range(self.length):
            self.deltaWarField[i] += stepFunction(abs(self.formerWarField[i] - warField[i]))
        self.formerWarField = warField
        return self.deltaWarField

