from TrafficPredictor.Probability import distributions


class BayesNet:
    def __init__(self):
        self.nodes = {}

    def _add_parent(self, distribution: distributions.Distribution) -> None:
        if distribution.name in self.nodes:
            raise ValueError(f"Node with name {distribution.name} already exists")
        self.nodes[distribution.name] = distribution

    def _add_child(self, conditional_distribution: distributions.ConditionalDistribution) -> None:
        if not conditional_distribution.is_complete():
            raise ValueError("The conditional distribution is not complete")
        parent_nodes = conditional_distribution.parent_names
        if not all(x in self.nodes for x in parent_nodes):
            raise ValueError("At least one specified parent node in list does not exist")
        self.nodes[conditional_distribution.name] = conditional_distribution

    def get_nodes(self) -> tuple:
        return tuple(self.nodes.keys())

    def get_node_values(self, node: str) -> tuple:
        return self.nodes[node].get_values()

    def query_format(self):
        query_format = {}
        for node in self.get_nodes():
            query_format[node] = []
            for val in self.get_node_values(node):
                query_format[node].append(val)
            query_format[node] = tuple(query_format[node])
        return query_format

    def add_node(self, node: distributions) -> None:
        if type(node) is distributions.Distribution:
            self._add_parent(node)
        else:
            assert type(node) is distributions.ConditionalDistribution
            self._add_child(node)

    def probability(self, observations: dict, uselog=True) -> float:
        from math import log
        if not set(observations.keys()) == set(self.nodes.keys()):
            raise ValueError(f"Entered observations must be exactly the same as the nodes in the network, "
                             f"expected keys {set(self.nodes.keys())}, given keys: {set(observations.keys())}")
        if uselog:
            total = 0
        else:
            total = 1

        for variable_name, variable_value in observations.items():
            distr = self.nodes[variable_name]
            if type(distr) is distributions.Distribution:  # has no parents - regular distribution
                prob = distr.p(variable_value)
            else:  # has at least one parent - conditional distribution
                assert type(distr) is distributions.ConditionalDistribution
                parents_value = [None] * len(distr.parent_names)
                for parent in distr.parent_names:
                    parents_value[distr.parent_names.index(parent)] = observations[parent]
                prob = distr.p(tuple(parents_value), variable_value)
            # Handling probability
            if prob == 0:
                if uselog:
                    return -float("inf")
                else:
                    return 0
            else:
                if uselog:
                    total += log(prob)
                else:
                    total *= prob

        return total

    def __repr__(self):
        me = ""
        for node in self.nodes:
            d = self.nodes[node]
            if type(d) is distributions.Distribution:  # has no parents
                me += f"\nNode: {node} (No Parents)"
            else:  # has to least one parent
                assert type(d) is distributions.ConditionalDistribution
                me += f"\nNode: {node} (Parents: {d.parent_names})"
        return me[1:]

    def __str__(self):
        return self.__repr__()
