from abc import ABC, abstractmethod


class Agent:
    @abstractmethod
    def get_move(self, vi, state_value, first_player):
        pass
