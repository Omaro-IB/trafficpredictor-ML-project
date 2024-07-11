from itertools import product


class Distribution:
    def __init__(self, name: str, values: tuple) -> None:
        self.name = name  # random variable name: str
        self._values = values  # random variable values it can take on: tuple of str

        self.size = len(values)  # # of entries in distribution table
        self._distr = {}  # variable value: variable probability pairs
        self._complete = 0
        # = 1 if values match the distribution
        # = 2 if distribution sum is 1
        # = 3 if both 1 and 2
        # = 0 if none

    def get_values(self):
        return self._values

    def get_actual_size(self) -> int:
        return len(self._distr)

    def is_complete(self) -> bool:
        return self._complete == 3

    def get_incomplete(self) -> tuple:
        return tuple(x for x in self._values if x not in self._distr)

    def _check_complete(self) -> None:
        self._complete = 0
        if (len(self._distr) == self.size) and (all([val in self._distr for val in self._values])):
            self._complete += 1
        if round(sum(self._distr[i] for i in self._distr), 4) == 1:
            self._complete += 2

    def set_p(self, value: str, probability: float) -> None:
        if value not in self._values:
            raise ValueError(f"Value {value} not in distribution {self._distr}")
        self._distr[value] = probability
        self._check_complete()

    def p(self, value: str, check=True) -> float:
        if check:
            if self._complete == 0:
                raise ValueError("Distribution probabilities do not sum to 1 AND are not all set")
            if self._complete == 1:
                raise ValueError("Distribution probabilities do not sum to 1")
            if self._complete == 2:
                raise ValueError("Distribution probabilities are not all set")
            assert self._complete == 3
        return self._distr[value]

    def __repr__(self) -> str:
        me = f"{self.name}\n============================="
        for v in self._values:
            try:
                prob = str(self._distr[v])
            except KeyError:
                prob = "-"
            me += f"\n{v}\t|\t{prob}"
        return me

    def __str__(self) -> str:
        return self.__repr__()


class ConditionalDistribution:
    def __init__(self, parent_names: tuple, parent_values: tuple, name: str, values: tuple):
        self.name = name  # random variable name: str
        self._values = values  # random variable values it can take on: tuple of str
        self.parent_names = parent_names  # random variable parent names: tuple of str
        self._parent_values = parent_values  # random variable values parents can take on: tuple of tuple of str

        self._combinations = tuple(product(*parent_values))
        self.size = len(values) * len(self._combinations)  # # of entries in conditional distribution table
        self._cdistr = {}  # parent values: Distribution pairs
        self._complete = 0
        # = 1 if values match the conditional distribution
        # = 2 if each sub-distribution is complete
        # = 3 if both 1 and 2
        # = 0 if none

    def get_values(self):
        return self._values

    def get_actual_size(self) -> int:
        actual_size = 0
        for d in self._cdistr:
            actual_size += self._cdistr[d].get_actual_size()
        return actual_size

    def is_complete(self) -> bool:
        return self._complete == 3

    def get_incomplete(self) -> tuple:
        incomplete = []
        for comb in self._combinations:
            x = self._cdistr[comb].get_incomplete()
            if x:
                incomplete.append((comb, x))
        return tuple(incomplete)

    def _check_complete(self) -> None:
        self._complete = 0
        actual_size = self.get_actual_size()
        if actual_size == self.size and all([parent in self._cdistr for parent in self._combinations]):
            self._complete += 1

        all_complete = True
        for d in self._cdistr:
            if not self._cdistr[d].is_complete():
                all_complete = False
                break
        if all_complete:
            self._complete += 2

    def set_p(self, parents_value: tuple, value: str, probability: float) -> None:
        if value not in self._values:
            raise ValueError(f"Value {value} not in distribution {self._values}")
        if parents_value not in self._combinations:
            raise ValueError("Parent values not in distribution")
        try:
            self._cdistr[parents_value].set_p(value, probability)
        except KeyError:
            self._cdistr[parents_value] = Distribution(f"{parents_value} Sub-Distribution", self._values)
            self._cdistr[parents_value].set_p(value, probability)
        self._check_complete()

    def p(self, parents_value: tuple, value: str, check=True) -> float:
        if check:
            if self._complete == 0:
                raise ValueError("Sub-distributions are not complete AND are not all set")
            if self._complete == 1:
                raise ValueError("At least one sub-distribution is not complete- make sure each sub-distribution sums to 1 and is fully set")
            if self._complete == 2:
                raise ValueError("At least one sub-distribution is not set")
            assert self._complete == 3
        return self._cdistr[parents_value].p(value)

    def __repr__(self) -> str:
        me = ""
        for pv in self._combinations:
            try:
                subd = str(self._cdistr[pv])
            except KeyError:
                subd = f"=============================\n{pv} Sub-Distribution Not Set"
            me += f"\n{subd}"
        return me[1:]

    def __str__(self) -> str:
        return self.__repr__()
