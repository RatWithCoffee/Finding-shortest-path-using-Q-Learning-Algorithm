import numpy as np
import pylab as pl

from undirected_graph import UndirectedGraph
from q_learning import QLearning


def run(gamma, alfa, num_of_iterations):
    graph = UndirectedGraph(edges, start_ver, goal_vert, num_of_vertices)
    graph.visualize_graph()

    q_learning = QLearning(graph, gamma, alfa)
    scores = q_learning.learning(num_of_iterations)
    shortest_path = q_learning.get_shortest_path()

    print("Самый короткий путь:")
    print(shortest_path)

    pl.plot(scores)
    pl.xlabel('Номер итерации')
    pl.ylabel('Значение Q')
    pl.show()


if __name__ == "__main__":
    edges = [(0, 1), (1, 5), (5, 6), (5, 4), (1, 2),
             (1, 3), (9, 10), (2, 4), (0, 6), (6, 7),
             (8, 9), (7, 8), (1, 7), (3, 9), (2, 10)]
    num_of_vertices = 11
    start_ver = 0
    goal_vert = 10
    gamma = 0.75
    num_of_iterations = 500
    alfa = 1
    print("Коэффициент обесценивания:", gamma)
    print("Скорость обучения:", alfa)
    print("Количество итераций:", num_of_iterations)
    run(gamma, alfa, num_of_iterations)
