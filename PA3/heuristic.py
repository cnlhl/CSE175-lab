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
import game
from parameters import *
from minimax import probability_of_time
from minimax import value

def expected_value_over_delays(state, ply):
    """Calculate the expected utility over all possible randomly selected
    Guardian delay times. Return this expected utility value."""
    val = 0.0
    # PLACE YOUR CODE HERE
    # Note that the value of "ply" must be passed along, without
    # modification, to any function calls that calculate the value 
    # of a state.
    for i in range(min_time_steps, max_time_steps + 1):
        tmp_state = copy.copy(state)
        tmp_state.time_remaining = i
        val += probability_of_time(i) * value(tmp_state, ply+1)
    return val

def heuristic_value(state):
    """Return an estimate of the expected payoff for the given state of
    game play without performing any look-ahead search. This value must
    be between the maximum payoff value and the additive inverse of the
    maximum payoff."""
    val = 0.0
    act = state.action
    A, B = 15, 10
    if state.need_time():
        risk_factor = 0
        for i in range(min_time_steps, max_time_steps + 1):
            if i > state.action:
                risk_factor += probability_of_time(i) * state.action
            else:
                risk_factor -= probability_of_time(i) * state.action
        if state.current_turn == Player.east:
            risk_factor = -risk_factor
        
    else:
        state.complete_turn()
        state.check_for_winner()
        if state.terminal_state():
            risk_factor = state.payoff()
        else:
            risk_factor = 0
            
    distance_factor = state.e_loc + state.w_loc
    val = A * distance_factor + B * risk_factor
    # print('action:',act,'distance_factor:', distance_factor, 'risk_factor:', risk_factor, 'heuristic_value:', val)
    
    return val