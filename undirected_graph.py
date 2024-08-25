import networkx as nx
import pylab as pl


class UndirectedGraph:

    def __init__(self, edges, start_vert, goal_vert, num_of_vertices):
        self.edges = edges
        self.goal_vert = goal_vert
        self.start_vert = start_vert
        self.num_of_vertices = num_of_vertices

    def visualize_graph(self):
        graph = nx.Graph()
        graph.add_edges_from(self.edges)
        pos = nx.spring_layout(graph)
        nx.draw_networkx_nodes(graph, pos)
        nx.draw_networkx_edges(graph, pos)
        nx.draw_networkx_labels(graph, pos)
        pl.show()
