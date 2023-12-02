#
# heuristic.py
#
# This Python script file provides two functions in support of minimax search
# using the expected value of game states. First, the file provides the
# function "expected_value_over_delays". This function takes as an argument
# a state of game play in which the current player has just selected an
# action. The function calculates the expected value of the state over all
# possible random results determining the amount of time before the
# Guardian changes gaze direction. This function calculates this value
# regardless of whose turn it is. The value of game states that result from
# different random outcomes is determined by calling "value". Second, the
# file provides a heuristic evaluation function for non-terminal game states.
# The heuristic value returned is between "max_payoff" (best for the
# computer player) and negative one times that value (best for the opponent).
# The heuristic function may be applied to any state of play. It uses
# features of the game state to predict the game payoff without performing
# any look-ahead search.
#
# This content is protected and may not be shared, uploaded, or distributed.
#
# PLACE ANY COMMENTS, INCLUDING ACKNOWLEDGMENTS, HERE
#
# PLACE YOUR NAME AND THE DATE HERE
# Haolin Li 2023-11-26

import copy
from parameters import *
from minimax import probability_of_time
from minimax import value

def expected_value_over_delays(state, ply):
    """Calculate the expected utility over all possible randomly selected
    Guardian delay times. Return this expected utility value."""
    val = 0.0
    # PLACE YOUR CODE HERE
    
    # estimate the expected value of the state over all possible random results
    # give different delay time with different value weight (probablity weighted average)
    for i in range(min_time_steps, max_time_steps + 1):
        tmp_state = copy.copy(state)
        tmp_state.time_remaining = i
        val += probability_of_time(i) * value(tmp_state, ply)
    return val

def heuristic_value(state):
    """Return an estimate of the expected payoff for the given state of
    game play without performing any look-ahead search. This value must
    be between the maximum payoff value and the additive inverse of the
    maximum payoff."""
    val = 0.0    
    
    # A, B, C, D are the weight for each factor
    # adjust the weight control the strategy of the computer player
    A,B,C,D = 20, 30, 20, 30
    
    w_distance = abs(state.w_loc)
    e_distance = abs(state.e_loc)
    
    # the closer to the center, the value of this two factor grows bigger
    # for the computer player, when it gets closer to the center, it's stragety can be more aggressive
    # for the opponent, when it gets closer to the center, it's stragety can be more defensive
    # reflected on the value, we set aggressive_factor to be positive, defensive_factor to be negative
    aggressive_factor = 1/w_distance
    defensive_factor = 1/e_distance
    
    # since e_loc is set to be positive, w_loc is set to be negative
    # the sum of them can reflect the relative distance between the two players
    # the bigger the sum, the better for the computer player (closer to the center)
    distance_factor = state.e_loc + state.w_loc
    
    # heuristic function is called within min_value_over_actions() and max_value_over_actions()
    # there're extra information(action steps) in state, which can be used to adjust the strategy
    # the bigger the action, the more risky the strategy is, when the current turn is the computer player
    # the smaller the risky factor, the better for the computer player
    # correspondingly, when the current turn is the opponent, the bigger the risky factor, the better for the computer player
    risk_factor = state.action/4
    if state.current_turn == Player.east:
        risk_factor = - risk_factor
    
    val = A * distance_factor + B * aggressive_factor - C * defensive_factor - D *risk_factor
    return val