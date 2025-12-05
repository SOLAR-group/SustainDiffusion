from enum import Enum


class Objectives(Enum):
    FAIRNESS = "fairness"
    ENERGY = "energy"
    ALL = "all"
    CPU = "cpu"
    GPU = "gpu"
    DURATION = "duration"
    IMAGE = "image"

class Fitness(Enum):
    NSGAII = "nsgaii"
    NSGAIII = "nsgaiii"
    WEIGHT = "weight"
