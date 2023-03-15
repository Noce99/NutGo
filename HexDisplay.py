import numpy as np
import pygame
import torch
from screeninfo import get_monitors
from pygame.locals import *
from math import sin, cos, pi, sqrt

from NutGo.Neural_Networks import to_cuda


class Visualizer:
    def __init__(self, board_size):
        pygame.init()
        monitor_height = get_monitors()[0].height
        self.window_size = monitor_height - int(0.2 * monitor_height)
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("Hex Game")
        self.board_size = board_size
        self.line_size = (self.window_size - self.window_size * 0.1) // (2 * sin(pi / 3) * (self.board_size - 1))
        self.screen.fill((255, 255, 158))
        self.line_width = 5
        self.circle_radius = 10
        self.color_red_line = (255, 0, 0)
        self.color_blue_line = (0, 0, 255)
        self.color_red_connected_left = (230, 255, 0)
        self.color_red_connected_right = (0, 255, 43)
        self.color_blue_connected_left = (0, 255, 239)
        self.color_blue_connected_right = (255, 0, 239)
        self.circles = [[None for _ in range(board_size)] for _ in range(board_size)]
        self.probability = None

    def print_board(self, board, first_player_turn):
        self.screen.fill((255, 255, 158))
        if first_player_turn is not None:
            red = (255, 0, 0)
            green = (0, 255, 0)
            blue = (0, 0, 128)
            sys_font = pygame.font.get_default_font()
            font = pygame.font.SysFont(sys_font, 30)
            if first_player_turn:
                text = font.render(" RED PLAYING ", True, red, green)
            else:
                text = font.render("BLUE PLAYING", True, blue, green)
            text_rect = text.get_rect()
            text_rect.center = (80, 15)
            self.screen.blit(text, text_rect)
        cx = self.window_size // 2
        cy = self.window_size * 0.05
        for r, row in enumerate(board):
            for c, element in enumerate(row):
                if c != self.board_size - 1:
                    if r == 0 or r == self.board_size - 1:
                        color = self.color_red_line
                    else:
                        color = (0, 0, 0)
                    pygame.draw.line(self.screen, color, (cx + c * self.line_size * cos(pi / 3),
                                                          cy + c * self.line_size * sin(pi / 3)),
                                     (cx + (c + 1) * self.line_size * cos(pi / 3),
                                      cy + (c + 1) * self.line_size * sin(pi / 3)),
                                     self.line_width)
                if r != self.board_size - 1:
                    if c == 0 or c == self.board_size - 1:
                        color = self.color_blue_line
                    else:
                        color = (0, 0, 0)
                    pygame.draw.line(self.screen, color, (cx + c * self.line_size * cos(pi / 3),
                                                          cy + c * self.line_size * sin(pi / 3)),
                                     (cx + c * self.line_size * cos(pi / 3) - self.line_size * cos(pi / 3),
                                      cy + c * self.line_size * sin(pi / 3) + self.line_size * sin(pi / 3)),
                                     self.line_width)
                if c != 0 and r != self.board_size - 1:
                    pygame.draw.line(self.screen, (0, 0, 0), (cx + c * self.line_size * cos(pi / 3),
                                                              cy + c * self.line_size * sin(pi / 3)),
                                     (cx + (c - 1) * self.line_size * cos(pi / 3) - self.line_size * cos(pi / 3),
                                      cy + (c - 1) * self.line_size * sin(pi / 3) + self.line_size * sin(pi / 3)),
                                     self.line_width)
                if element is None:
                    pygame.draw.circle(self.screen, (0, 0, 0), (cx + c * self.line_size * cos(pi / 3),
                                                                cy + c * self.line_size * sin(pi / 3)),
                                       self.circle_radius)
                    self.circles[r][c] = (cx + c * self.line_size * cos(pi / 3), cy + c * self.line_size * sin(pi / 3))
                else:
                    self.circles[r][c] = None
                    if element[1] is True:
                        if element[0] is True:
                            pygame.draw.circle(self.screen, self.color_red_connected_left,
                                               (cx + c * self.line_size * cos(pi / 3),
                                                cy + c * self.line_size * sin(pi / 3)), self.circle_radius + 2)
                        else:
                            pygame.draw.circle(self.screen, self.color_blue_connected_left,
                                               (cx + c * self.line_size * cos(pi / 3),
                                                cy + c * self.line_size * sin(pi / 3)), self.circle_radius + 2)
                    elif element[1] is False:
                        if element[0] is True:
                            pygame.draw.circle(self.screen, self.color_red_connected_right,
                                               (cx + c * self.line_size * cos(pi / 3),
                                                cy + c * self.line_size * sin(pi / 3)), self.circle_radius + 2)
                        else:
                            pygame.draw.circle(self.screen, self.color_blue_connected_right,
                                               (cx + c * self.line_size * cos(pi / 3),
                                                cy + c * self.line_size * sin(pi / 3)), self.circle_radius + 2)
                    if element[0] is True:
                        pygame.draw.circle(self.screen, (255, 0, 0),
                                           (cx + c * self.line_size * cos(pi / 3),
                                            cy + c * self.line_size * sin(pi / 3)), self.circle_radius)
                    else:
                        pygame.draw.circle(self.screen, (0, 0, 255),
                                           (cx + c * self.line_size * cos(pi / 3),
                                            cy + c * self.line_size * sin(pi / 3)), self.circle_radius)

            cx -= self.line_size * cos(pi / 3)
            cy += self.line_size * sin(pi / 3)
        if self.probability is not None:
            self.print_probability(self.probability)
        pygame.display.flip()

    def print_probability(self, probability, offset=0):
        self.probability = probability
        red = (255, 0, 0)
        green = (0, 255, 0)
        blue = (0, 0, 128)
        sys_font = pygame.font.get_default_font()
        font = pygame.font.SysFont(sys_font, 30)
        cx = self.window_size // 2
        cy = self.window_size * 0.05
        i_max = torch.argmax(probability)
        for r in range(self.board_size):
            for c in range(self.board_size):
                if r * self.board_size + c == i_max:
                    text = font.render(f"{float(probability[r * self.board_size + c]):.2f}", True, red, blue)
                else:
                    text = font.render(f"{float(probability[r * self.board_size + c]):.2f}", True, green, blue)
                text_rect = text.get_rect()
                text_rect.center = (self.circle_radius * 3 + cx + c * self.line_size * cos(pi / 3),
                                    offset + cy + c * self.line_size * sin(pi / 3))
                self.screen.blit(text, text_rect)
            cx -= self.line_size * cos(pi / 3)
            cy += self.line_size * sin(pi / 3)
        pygame.display.flip()

    def check_if_is_clicking_a_circle(self, pos):
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.circles[r][c] is not None:
                    if sqrt((pos[0]-self.circles[r][c][0])**2 + (pos[1]-self.circles[r][c][1])**2) < self.circle_radius:
                        return r, c
        return None

    def human_turn(self, board):
        done = False
        # self.print_board(board, None)
        while not done:
            event_list = pygame.event.get()
            for ev in event_list:
                if ev.type == pygame.QUIT:
                    exit()
                elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                    clicked_circle = self.check_if_is_clicking_a_circle(ev.pos)
                    if clicked_circle is not None:
                        return clicked_circle

    def finished_game(self, board):
        done = False
        self.print_board(board, None)
        pygame.display.flip()
        while not done:
            event_list = pygame.event.get()
            for ev in event_list:
                if ev.type == pygame.QUIT:
                    exit()

    @staticmethod
    def wait_until_click():
        done = False
        while not done:
            event_list = pygame.event.get()
            for ev in event_list:
                if ev.type == pygame.QUIT:
                    exit()
                elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                    done = True

    def show_dataset(self, dataset):
        for i in range(len(dataset)):
            for j in range(dataset[i][0].shape[0]):
                state = dataset[i][0][j]
                result = dataset[i][1][j]
                hex_state = [[None for _ in range(self.board_size)] for _ in range(self.board_size)]
                r = 0
                c = 0
                for ii, element in enumerate(state[1:]):
                    if element == 1:
                        hex_state[r][c] = (True, None)
                    elif element == -1:
                        hex_state[r][c] = (False, None)
                    c += 1
                    if (ii + 1) % self.board_size == 0:
                        r += 1
                        c = 0
                if state[0] == 1:
                    self.print_board(hex_state, True)
                else:
                    self.print_board(hex_state, False)
                self.print_probability(result)
                self.wait_until_click()
            break

    def visual_evaluation(self, dataset, nn):
        for i in range(len(dataset)):
            for j in range(dataset[i][0].shape[0]):
                state = dataset[i][0][j]
                result = dataset[i][1][j]
                hex_state = [[None for _ in range(self.board_size)] for _ in range(self.board_size)]
                r = 0
                c = 0
                for ii, element in enumerate(state[1:]):
                    if element == 1:
                        hex_state[r][c] = (True, None)
                    elif element == -1:
                        hex_state[r][c] = (False, None)
                    c += 1
                    if (ii + 1) % self.board_size == 0:
                        r += 1
                        c = 0
                if state[0] == 1:
                    self.print_board(hex_state, True)
                else:
                    self.print_board(hex_state, False)
                self.print_probability(result)
                predicted_tensor = torch.flatten(nn.predict(to_cuda(state[None, :])))
                self.print_probability(predicted_tensor, offset=30)
                self.wait_until_click()
            break

"""
sample_board = [[None for _ in range(10)] for _ in range(10)]
vi = Visualizer(10)
while True:
    clicked_circle = vi.human_turn(sample_board)
    sample_board[clicked_circle[0]][clicked_circle[1]] = (True, None)
    clicked_circle = vi.human_turn(sample_board)
    sample_board[clicked_circle[0]][clicked_circle[1]] = (False, None)
"""