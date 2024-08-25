import numpy as np
from networkx import edges


class QLearning:
    def __init__(self, graph, gamma, alfa):
        self.R = self.init_reward_matrix(graph.num_of_vertices, graph.goal_vert, graph.edges)  # Инициализация матрицы вознаграждений
        self.Q = np.matrix(np.zeros([graph.num_of_vertices, graph.num_of_vertices]))  # Инициализация Q-матрицы
        self.graph = graph
        self.gamma = gamma  # discount factor - коэффициент дисконтирования
        self.alfa = alfa  # learning rate -  коэффициент обучения

    def update_Q(self, current_state, action):
        """
       Обновляет значение Q для текущего состояния и действия.

       param current_state: текущее состояние
       param action: выполненное действие
       return: текущее значение награды
       """
        # Поиск максимального значения Q для следующего состояния
        max_index = np.where(self.Q[action,] == np.max(self.Q[action,]))[1]
        if max_index.shape[0] > 1:
            rng = np.random.default_rng()
            max_index = int(rng.choice(max_index))
        else:
            max_index = max_index.item()
        max_value = self.Q[action, max_index]

        # Обновление значения Q по уравнению Беллмана
        self.Q[current_state, action] = ((1 - self.alfa) * self.Q[current_state, action] +
                                         self.alfa * (self.R[current_state, action] + self.gamma * max_value))

        print(self.Q)
        # Получение суммы нормализованных значений Q для оценки обучения
        if np.max(self.Q) > 0:
            return np.sum(self.Q / np.max(self.Q) * 100)
        else:
            return 0

    def get_available_actions(self, state):
        """
            Возвращает доступные действия для заданного состояния.

            param state: текущее состояние
            return: список доступных действий
        """
        current_state_row = self.R[state,]
        return np.where(current_state_row >= 0)[1]

    def learning(self, num_of_iterations):
        """
            Основной цикл обучения Q-learning.

            param num_of_iterations: число итераций обучения
            return: список оценок на каждой итерации
        """
        scores = []
        for i in range(num_of_iterations):
            curr_vert = np.random.randint(0, int(self.Q.shape[0]))
            available_action = self.get_available_actions(curr_vert)
            action = self.get_random_next_action(available_action)
            score = self.update_Q(curr_vert, action)
            scores.append(score)
        return scores

    def get_shortest_path(self):
        """
            Возвращает кратчайший путь от начальной вершины до целевой вершины.

            return: список вершин, представляющий кратчайший путь
        """
        current_state = self.graph.start_vert
        steps = [current_state]

        while current_state != self.graph.goal_vert:
            next_step_index = np.where(self.Q[current_state,] == np.max(self.Q[current_state,]))[1]
            if next_step_index.shape[0] > 1:
                next_step_index = int(np.random.choice(next_step_index, size=1))
            else:
                next_step_index = next_step_index.item()
            steps.append(next_step_index)
            current_state = next_step_index

        return steps

    @staticmethod
    def get_random_next_action(available_action):
        """
            Выбирает случайное следующее действие из доступных.

            param available_action: список доступных действий
            return: выбранное действие
        """
        rng = np.random.default_rng()
        next_action = int(rng.choice(available_action))
        return next_action

    @staticmethod
    def init_reward_matrix(num_of_vertices, goal_vert, edges):
        """
            Инициализирует матрицу вознаграждений.

            param num_of_vertices: количество вершин в графе
            param goal_vert: целевая вершина
            return: матрица вознаграждений
        """
        r = np.matrix(np.ones(shape=(num_of_vertices, num_of_vertices)))
        r *= -1

        for point in edges:
            if point[1] == goal_vert:
                r[point] = 100
            else:
                r[point] = 0

            if point[0] == goal_vert:
                r[point[::-1]] = 100
            else:
                r[point[::-1]] = 0

        r[goal_vert, goal_vert] = 100
        return r
