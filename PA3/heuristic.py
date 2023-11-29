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
    
    # this procedure also expands the tree
    # if we don't increase the ply, the tree will be too large and the program will be too slow
    # but if we increase the ply every time, the tree will be too small
    # so we increase the ply only when it is east's turn
    if state.current_turn == Player.east:
        ply += 1
    
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
    
    # A and B are the weight of distance and risk factor
    # adjust the weight to make startegy more aggressive or conservative
    A, B = 12, 10
    
    # if time is needed, which means the parent layer is min/max layer
    # estimate the expexted foward steps(risk_factor) by taking the time probability weighted average
    if state.need_time():
        risk_factor = 0
        for i in range(min_time_steps, max_time_steps + 1):
            if i > state.action:
                risk_factor += probability_of_time(i) * state.action
            else:
                risk_factor -= probability_of_time(i) * state.action
        if state.current_turn == Player.east:
        # if it is east's turn, the risk factor should be reversed
            risk_factor = -risk_factor
        
    else:
    # if no time is needed, which means the parent layer is delay layer(expected_value_over_delays)
    # in this case, action is set by min/max layer, and time is set by delay layer
    # so we can directly update the state to the next state, and calculate the heuristic value 
    # by recursively calling heuristic_value to enter the first if statement
        return value(state,max_ply-1)
      
    # distance_factor finds the sum of the distance of two players
    # because the west player distance is negative, east player distance is positive
    # if west player is closer to the treasure, the distance_factor will be positive
    # if east player is closer to the treasure, the distance_factor will be negative      
    distance_factor = state.e_loc + state.w_loc
    
    val = A * distance_factor + B * risk_factor
    
    return val