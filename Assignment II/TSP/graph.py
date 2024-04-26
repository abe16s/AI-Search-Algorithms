from typing import Hashable


class ReadOnly:
    def __init__(self, obj):
        self.__obj = obj

    def __getitem__(self, key):
        return ReadOnly(self.__obj[key])

    def __iter__(self):
        return iter(self.__obj)

    def __len__(self):
        return len(self.__obj)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return str(self.__obj)


class DirectedGraph:
    def __init__(self):
        self._graph = {}
        self.__readonly = ReadOnly(self._graph)

    @property
    def graph(self):
        return self.__readonly

    @graph.setter
    def graph(self, _):
        raise AttributeError(
            "Can't set content of graph directly. Use graph methods.")

    @property
    def nodes(self):
        return list(self._graph.keys())

    def add_node(self, node: Hashable) -> None:
        """Adds node to graph if node doesn't exist."""

        if node in self._graph:
            return

        self._graph[node] = []

    def add_edge(self, node_1: Hashable, node_2: Hashable, weight: float = 1) -> None:
        """"Add weighted directed edge from node_1 to node_2

        Creates node if the nodes does not exist.
        """

        self.add_node(node_1)
        self.add_node(node_2)

        self._graph[node_1].append((node_2, weight))

    def delete_edge(self, node_1: Hashable, node_2: Hashable) -> None:
        """Removes directed edge from node_1 to node_2"""

        neighbours = self._graph[node_1]

        write_ptr = 0
        for read_ptr in range(len(neighbours)):
            neighbour_node, _ = neighbours[read_ptr]

            if neighbour_node != node_2:
                neighbours[write_ptr] = neighbours[read_ptr]
                write_ptr += 1

        del self._graph[node_1][write_ptr:]

    def delete_node(self, node: Hashable) -> None:
        """Delete node from graph, removing corresponding edges."""

        for other_node in self.nodes:
            self.delete_edge(other_node, node)

        del self._graph[node]

    def neighbours(self, node: Hashable) -> ReadOnly:
        """Returns ReadOnly list containing neighbours of node"""

        return ReadOnly(self._graph[node])

    def __str__(self) -> str:
        return str(self._graph)

    def __iter__(self):
        for node in self.nodes:
            yield node

    def __len__(self):
        return len(self._graph)
    __repr__ = __str__


class UndirectedGraph(DirectedGraph):
    def __init__(self):
        super().__init__()

    def add_edge(self, node_1: Hashable, node_2: Hashable, weight: float = 1) -> None:
        super().add_edge(node_1, node_2, weight)
        super().add_edge(node_2, node_1, weight)

    def delete_edge(self, node_1: Hashable, node_2: Hashable) -> None:
        super().delete_edge(node_1, node_2)
        super().delete_edge(node_2, node_1)
