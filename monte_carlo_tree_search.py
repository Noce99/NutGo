from abc import ABC, abstractmethod
from math import sqrt, log
import time

EXPLORATION_CONSTANT = 1


class MCTS:
    """Monte Carlo tree searcher."""

    def __init__(self, root_node, explore_time_limit):
        """
        :param root_node: Simply the root node
        :param explore_time_limit: The maximum explore time limit in seconds
        """
        self.root_node = root_node
        self.explore_time_limit = explore_time_limit

        self.path = [self.root_node]
        self.actual_node = self.root_node
        self.root_node.explore()

    def explore(self, verbose):
        if verbose:
            print("Starting Exploration!")
            print("Actual Node:")
            print(self.actual_node)
        starting_time = time.perf_counter()
        while time.perf_counter() - starting_time < self.explore_time_limit:
            if verbose:
                print("@" * 30)
            next_child = self.actual_node.get_next_unexplored_child()
            if next_child is not None:
                # Actual Node have some not explored child: next_child
                if verbose:
                    print("Find out a non explored child: ")
                    print(next_child)
                next_child.explore()
                self.path.append(next_child)
                first_player_won = next_child.roll_out()
                if verbose:
                    print(f"After Roll_Out the winner is first player? {first_player_won}")
                for node in self.path:
                    node.update_statistic(first_player_won)
                if verbose:
                    print("After having updated statistic this are the child:")
                    self.actual_node.print_children(0, deep=1)
                self.path = [self.root_node]
                self.actual_node = self.root_node
            else:
                # Actual Node have all the children already explored
                next_child = self.actual_node.get_best_explored_child()
                if next_child is None:
                    # Actual Node is a Terminal Node
                    first_player_won = self.actual_node.get_who_won()
                    if verbose:
                        print("The current node is final:")
                        print(self.actual_node)
                        print(f"And the winner is first player? {first_player_won}")
                    for node in self.path:
                        node.update_statistic(first_player_won)
                    if verbose:
                        print("After having updated statistic this are the child:")
                        self.actual_node.print_children(0, deep=1)
                    self.actual_node = self.root_node
                    self.path = [self.root_node]
                else:
                    # Actual Node have a child: next_child
                    if verbose:
                        print("Selecting an explored child:")
                        print(next_child)
                        print("Chosen between:")
                        self.actual_node.print_children(0, deep=1)
                    self.path.append(next_child)
                    self.actual_node = next_child
        print(f"I have explored for {time.perf_counter() - starting_time} seconds!")
        # self.root_node.print_children(0, deep=2)
        result = [(child, child.wins / child.N) for child in self.root_node.children]
        return result


class Node(ABC):
    """
    A representation of a single board state.
    """

    def __init__(self, first_player: bool, state_value):
        """
        :param first_player: True if the next one to move is the first player, False otherwise
        :param state_value: A unique value that represent the state
        """
        self.first_player = first_player
        self.state_value = state_value

        self.children = []
        self.explored = False
        self.all_children_explored = False
        self.terminal_Node = None
        self.N = 0
        self.wins = 0
        self.uct = None
        self.u = None

    def print_children(self, indentation, deep):
        """
        :param indentation: The number of tub before the output
        :param deep: Number of children layers
        :return:
        """
        print("\t" * indentation, end="")
        print(f"{[str(self)]} N = {self.N} wins = {self.wins} fp = {self.first_player} value = {self.uct} " +
              f"u = {self.u}")
        if deep > 0:
            for child in self.children:
                child.print_children(indentation + 1, deep - 1)

    def explore(self):
        self.terminal_Node = self.check_if_terminal()
        if not self.terminal_Node:
            self.find_children()
            if len(self.children) == 0:
                raise "Can't find children of a non terminal Node!"
        self.explored = True

    def update_statistic(self, first_player_won: bool):
        """
        :param first_player_won: True if the current player win, False if it loses and None if they tie
        """
        self.N += 1
        if first_player_won is not None:
            if first_player_won:
                self.wins += 1
            else:
                self.wins -= 1

    def is_terminal(self):
        """
        :return: True if the node is terminal
        """
        if not self.explored:
            return self.check_if_terminal()
        return self.terminal_Node

    def is_explored(self):
        """
        :return: True if the node was already explored.
        Explored means that we have already calculated all the possible children.
        """
        return self.explored

    def is_first_player(self):
        """
        :return: True if the first player is the next to move
        """
        return self.first_player

    def get_next_unexplored_child(self):
        """
        :return: return the next non explored child, "None" if all the children where explored,
                    raise an error if the current node was not explored
        """
        if not self.explored:
            raise "You asked an unexplored children to a non explored Node!"
        if self.all_children_explored:
            return None
        for child in self.children:
            if not child.explored:
                return child
        self.all_children_explored = True
        return None

    def get_best_explored_child(self):
        """
        :return: return the explored child with the highest or lower value, "None" if this node does not have
                    any children, rise an error if it was not explored and also rise an error
                    if not all the children were explored
        """
        if not self.explored:
            raise "You asked the best explored child to a non explored Node!"
        if not self.all_children_explored:
            raise "You asked the best child to a Node with some unexplored children"
        if self.terminal_Node:
            return None
        best_child = None
        best_child_uct = None
        if self.is_first_player():
            for child in self.children:
                child.uct = child.wins/child.N + EXPLORATION_CONSTANT * sqrt(log(self.N) / (1 + child.N))
                # child.u = + EXPLORATION_CONSTANT * sqrt(log(self.N) / (1 + child.N))
                if best_child is None:
                    best_child = child
                    best_child_uct = child.uct
                elif child.uct > best_child_uct:
                    best_child = child
                    best_child_uct = child.uct
        else:
            for child in self.children:
                child.uct = child.wins/child.N - EXPLORATION_CONSTANT * sqrt(log(self.N) / (1 + child.N))
                # child.u = - EXPLORATION_CONSTANT * sqrt(log(self.N) / (1 + child.N))
                if best_child is None:
                    best_child = child
                    best_child_uct = child.uct
                elif child.uct < best_child_uct:
                    best_child = child
                    best_child_uct = child.uct
        if best_child is None:
            print(f"Terminal_Node : {self.terminal_Node}")
            print(f"Explored : {self.explored}")
            print(f"Children : {self.children}")
            raise f"Seems like the Node is not terminal but it also don't have any child!"
        return best_child

    def get_children(self):
        """
        :return: "self.children" if explored otherwise raise an error
        """
        if not self.explored:
            raise "You asked children to a non explored Node!"
        return self.children

    def add_children(self, child_node):
        self.children.append(child_node)

    def roll_out(self):
        """
        The system play random move till the end of the game starting from this Node
        :return: True if first_player won, False if the second player won and None in case of tie
        """
        actual_node = self
        while True:
            if actual_node.is_terminal():
                return actual_node.get_who_won()
            actual_node = actual_node.get_random_child()

    def __str__(self):
        return self.state_value

    @abstractmethod
    def find_children(self):
        """
        Find all possible children of this node
        You need to use the "self.add_children" method
        """
        raise "You need to Implement 'find_children'!"

    @abstractmethod
    def get_random_child(self):
        """
        Find a random child of this Node
        :return: A Node that is a child of "self" and None if is not possible to find a child
        """
        raise "You need to Implement 'get_random_child'!"

    @abstractmethod
    def get_who_won(self):
        """
        Get who won this game
        :return: True if first_player won, False if the second player won and None in case of tie or in case none won
                    (it's not a terminal state)
        """
        raise "You need to Implement 'get_who_won'!"

    @abstractmethod
    def check_if_terminal(self):
        """
        Check if the current state is a final one
        :return: True if the state is final False otherwise
        """
        raise "You need to Implement 'check_if_terminal'!"
