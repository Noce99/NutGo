import random
from abc import ABC, abstractmethod
from math import sqrt, log
import time


class MCTS:
    """Monte Carlo tree searcher."""

    def __init__(self, root_node, explore_time_limit):
        """
        :param root_node: Simply the root node
        :param explore_time_limit: The maximum explore time limit in seconds
        """
        self.root_node = root_node
        self.root_node.explore()
        self.path = []
        self.actual_node = self.root_node
        self.explore_time_limit = explore_time_limit

    def explore(self):
        starting_time = time.perf_counter()
        while time.perf_counter() - starting_time < self.explore_time_limit:
            next_child = self.actual_node.get_next_unexplored_child
            if next_child is None:
                # All the children where already explored
                next_child = self.actual_node.get_best_explored_child()
                if next_child is None:
                    break
                self.path.append(next_child)
                self.actual_node = next_child
            else:
                next_child.explore()
                self.path.append(next_child)
                founded_result = self.actual_node.roll_out(next_child)


class Node(ABC):
    """
    A representation of a single board state.
    """

    def __init__(self, first_player: bool, state_value):
        """
        :param first_player: True if is a Node for the first player, False if is a Node for the second one
        :param state_value: A unique value that represent the state
        """
        self.children = []
        self.children_counter = 0
        self.state_value = state_value
        self.explored = False
        self.first_player = first_player
        self.all_children_explored = False
        self.N = None
        self.Ns = None
        self.Qs = None

    def explore(self):
        self.__find_children()
        self.explored = True
        self.N = 0
        self.Ns = [0 for _ in range(len(self.children))]
        self.Qs = [0 for _ in range(len(self.children))]

    def update_statistic(self, first_player_win: bool, children_index: int):
        """
        :param first_player_win: True if the current player win, False if it loses and None if they tie
        :param children_index: The index of the children that have been used to explore
        """
        self.N += 1
        self.Ns[children_index] += 1
        if first_player_win is None:
            self.Qs[children_index] += (0.5 - self.Qs[children_index]) / (self.Ns[children_index])
        elif first_player_win:
            if self.first_player:
                self.Qs[children_index] += (1 - self.Qs[children_index]) / (self.Ns[children_index])
            else:
                self.Qs[children_index] += (0 - self.Qs[children_index]) / (self.Ns[children_index])
        else:
            if self.first_player:
                self.Qs[children_index] += (0 - self.Qs[children_index]) / (self.Ns[children_index])
            else:
                self.Qs[children_index] += (1 - self.Qs[children_index]) / (self.Ns[children_index])

    @abstractmethod
    def __find_children(self):
        """
        Find all possible children of this node
        You need to use the "self.add_children" method
        """
        pass

    @abstractmethod
    def __get_random_child(self):
        """
        Find a random child of this Node
        :return: A Node that is a child of "self" and None if is not possible to find a child
        """
        pass

    @abstractmethod
    def __get_who_won(self):
        """
        Get who won this game (the state should be a final one)
        :return: True if first_player won, False if the second player won and None in case of tie
        """
        pass

    def is_terminal(self):
        """
        :return: True if the node has no children
        """
        if not self.explored:
            return None
        if len(self.children) == 0:
            return True
        else:
            return False

    def is_explored(self):
        """
        :return: True if the node was already explored.
        Explored means that we have already calculated all the possible children.
        """
        return self.explored

    def is_first_player(self):
        """
        :return: True if is the First Player turn
        """
        return self.first_player

    def get_next_unexplored_child(self):
        """
        :return: return the next non explored child, "None" if all the children where explored,
                    rais an error if the current node was not explored
        """
        if not self.explored:
            raise "You asked children to a non explored Node!"
        if self.all_children_explored:
            return None
        for _, c in self.children:
            if not c.explored:
                return c
        self.all_children_explored = True
        return None

    def get_best_explored_child(self, exploration_constant):
        """
        :return: return the explored child with the highest value, "None" if this node does not have
                    any children, rise an error if it was not explored and also rise an error
                    if not all the children were explored
        """
        if not self.explored:
            raise "You asked children to a non explored Node!"
        if not self.all_children_explored:
            raise "You asked best child to a Node with some unexplored children"
        best_child = None
        best_child_value = None
        for i in range(len(self.children)):
            if self.first_player:
                value = self.Qs[i] + exploration_constant*sqrt((log(self.N))/(self.Ns[i]))
                if best_child is None:
                    best_child = i
                    best_child_value = value
                elif  value >  best_child_value:
                    best_child = i
                    best_child_value = value
            else:
                value = self.Qs[i] - exploration_constant * sqrt((log(self.N)) / (self.Ns[i]))
                if best_child is None:
                    best_child = i
                    best_child_value = value
                elif value < best_child_value:
                    best_child = i
                    best_child_value = value
        return self.children[best_child]

    def get_children(self):
        """
        :return: "self.children" if explored otherwise rais an error
        """
        if not self.explored:
            raise "You asked children to a non explored Node!"
        return [c for _, c in self.children]

    def add_children(self, children_state_value):
        self.children.append((self.children_counter, children_state_value))
        self.children_counter += 1

    def roll_out(self, starting_node):
        """
        The system play random move till the end of the game
        :param starting_node: The Node from witch we are starting playing random moves
        :return: True if first_player won, False if the second player won and None in case of tie
        """
        actual_node = starting_node
        while True:
            next_node = actual_node.__get_random_child()
            if next_node is None:
                # actual_node is a final Node
                break
        return actual_node.__get_who_won()


    def __eq__(self, node2):
        return self.state_value == node2.state_value

    def __str__(self):
        return self.state_value


class TicTacToeNode(Node):

    def __init__(self, first_player: bool, state_value):
        """
        :param first_player: True if is a Node for the first player, False if is a Node for the second one
            we arbitrary decide that the first player is "x"
        :param state_value: A unique value that represent the state:
            x|_|o
            _|x|x   --> ["x", "_", "o", "_", "x", "x", "o", "o", "_"]
            o|o|_
        """
        super().__init__(first_player, state_value)

    def _Node__find_children(self):
        for i, c in enumerate(self.state_value):
            if c == "_":
                new_value = self.state_value[:]
                if self.is_first_player:
                    # x turn
                    new_value[i] = "x"
                    self.add_children(TicTacToeNode(first_player=False, state_value=new_value))
                else:
                    # o turn
                    new_value[i] = "o"
                    self.add_children(TicTacToeNode(first_player=True, state_value=new_value))

    def _Node__get_random_child(self):
        empty_places = [i for i, c in enumerate(self.state_value) if c == "_"]
        if empty_places:
            # "empty_place" is empty
            return None
        new_value = self.state_value[:]
        if self.is_first_player:
            new_value[random.choice(empty_places)] = "x"
        else:
            new_value[random.choice(empty_places)] = "o"
        return TicTacToeNode(first_player=not self.is_first_player, state_value=new_value)

    def _Node__get_who_won(self):
        # check raw
        for raw in range(3):
            first_el = self.state_value[raw*3]
            if self.state_value[raw*3+1] == first_el and self.state_value[raw*3+2] == first_el:
                # someone won
                if first_el == "x":
                    return True
                elif first_el == "o":
                    return False
                raise "Checking who won of a non final game!"
        # check columns
        for column in range(3):
            first_el = self.state_value[column]
            if self.state_value[column+3] == first_el and self.state_value[column+6] == first_el:
                # someone won
                if first_el == "x":
                    return True
                elif first_el == "o":
                    return False
                raise "Checking who won of a non final game!"
        # check diagonal
        first_el = self.state_value[0]
        if self.state_value[4] == first_el and self.state_value[8] == first_el:
            # someone won
            if first_el == "x":
                return True
            elif first_el == "o":
                return False
            raise "Checking who won of a non final game!"
        first_el = self.state_value[2]
        if self.state_value[4] == first_el and self.state_value[6] == first_el:
            # someone won
            if first_el == "x":
                return True
            elif first_el == "o":
                return False
            raise "Checking who won of a non final game!"
        return None

    def __str__(self):
        result = ""
        for i in range(3):
            for ii, c in enumerate(self.state_value[i*3:(i+1)*3]):
                if ii != 2:
                    result += f"{c}|"
                else:
                    result += f"{c}\n"
        return result


A = TicTacToeNode(False, ["x", "_", "o", "_", "x", "x", "o", "o", "_"])
print(A)
print("Their children are:")
A.explore()
for i, c in enumerate(A.get_children()):
    print(f"Children number {i}")
    print(c)
