from typing import List, Tuple

import numpy as np


class Diagram:
    def __init__(self, dim, values: List[Tuple] = None):
        if values is None:
            self.pd = []
        else:
            self.pd = values
        self.essential_points = [p for p in self.pd if p[1] == float('inf')]
        self.not_essential_points = [p for p in self.pd if p[1] != float('inf')]
        self.dim = dim

    def __getitem__(self, item):
        return self.pd[item]

    def __len__(self):
        return len(self.pd)

    def to_list(self):
        return self.pd

    @property
    def all_points(self):
        return self.essential_points + self.not_essential_points

    def normalize(self):
        if len(self.pd) == 0:
            return np.array([]), np.array([])

        min_birth = min([p[0] for p in self.pd])
        max_death = max([
            p[1] for p in self.not_essential_points]) if len(self.not_essential_points) != 0 \
            else max([p[0] for p in self.pd])

        norm_fact = max_death - min_birth or 1

        self.not_essential_points = [[
            (p[0] - min_birth) / norm_fact,
            (p[1] - min_birth) / norm_fact
        ] for p in self.not_essential_points]

        self.essential_points = [[(p[0] - min_birth) / norm_fact, 1] for p in self.essential_points]

    def threshold(self, t: float = 0.01):
        self.essential_points = list(p for p in self.essential_points if p[1] - p[0] > t)
        self.not_essential_points = list(p for p in self.not_essential_points if p[1] - p[0] > t)

    def __str__(self):
        return "<Diagram dim: %d, no. pairs: %d>" % (self.dim, len(self))


class PersistenceDiagrams:
    def __init__(self):
        self._diagrams = []

    @property
    def max_dim(self):
        return len(self._diagrams) - 1

    def __len__(self):
        return len(self._diagrams)

    def __getitem__(self, item) -> Diagram:
        return self._diagrams[item]

    @property
    def diagrams(self) -> List[Diagram]:
        return self._diagrams

    def add_diagram(self, values, dim=None):
        if dim is None:
            self._diagrams.append(Diagram(len(self), values))
        else:
            if dim > self.max_dim:
                assert dim == self.max_dim + 1
                self.diagrams.append(Diagram(dim, values))
            else:
                self._diagrams[dim] = Diagram(dim, values)

    @classmethod
    def from_list(cls, diagrams: List[List[Tuple[float, float]]]):
        d = cls()
        for dgm in diagrams:
            d.add_diagram(dgm)
        return d

    def to_list(self):
        return [d.to_list() for d in self.diagrams]

    def normalize(self):
        for d in self.diagrams:
            d.normalize()

    def threshold(self, t: float = 0.01):
        for d in self.diagrams:
            d.threshold(t)

    def __str__(self):
        return "<PersistenceDiagrams dim=%d>" % (len(self))
