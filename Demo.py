import time

from Agent_DRL import HexAgent
from Game_Player import HexGame


class TOPP:

    def __init__(self, board_size, total_episodes=200, m=5, g=2):
        self.board_size = board_size
        self.total_episodes = total_episodes
        self.m = m
        self.g = g

    def train(self):
        dlr_agent = HexAgent(board_size=self.board_size, batch_size=32, learning_rate=3, max_num_of_data=1000)
        for i in range(self.m):
            dlr_agent.train_while_playing(epochs=100, time_limit=2, simulations_limit=1000,
                                          num_of_games_before_evaluation=self.total_episodes // self.m,
                                          prob_of_random_move=0.8, do_evaluation=False, save_dataset=False,
                                          max_num_of_games=self.total_episodes // self.m)
            dlr_agent.evaluate_model()
            time.sleep(0.1)
            dlr_agent.save_weight(folder="./demo", name=f"player_{i}")

    def play_a_match(self, agent_1, agent_2):
        half_g = self.g // 2
        agent_1_wins = 0
        agent_2_wins = 0
        for i in range(half_g):
            a_game = HexGame(self.board_size)
            result = a_game.play_a_match(agent_1, agent_2, wait=False, verbose=False, visualizer=False)
            if result:
                agent_1_wins += 1
            else:
                agent_2_wins += 1
        for i in range(half_g):
            a_game = HexGame(self.board_size)
            result = a_game.play_a_match(agent_2, agent_1, wait=False, verbose=False, visualizer=False)
            if result:
                agent_2_wins += 1
            else:
                agent_1_wins += 1
        return agent_1_wins, agent_2_wins

    def play(self):
        players = [HexAgent(board_size=self.board_size, batch_size=32, learning_rate=3, max_num_of_data=1000)
                   for _ in range(self.m)]
        num_of_wins = [0 for _ in range(self.m)]
        for i in range(self.m):
            players[i].load_weight(f"{self.board_size}x{self.board_size}_player_{i}", folder="./demo")
        for i in range(self.m - 1):
            for j in range(i+1, self.m):
                i_wins, j_wins = self.play_a_match(players[i], players[j])
                num_of_wins[i] += i_wins
                num_of_wins[j] += j_wins
        for i in range(self.m):
            print(f"player_{i} won {num_of_wins[i]} times.")


if __name__ == "__main__":
    tournament = TOPP(board_size=5, total_episodes=25)
    tournament.train()
    #tournament.play()
