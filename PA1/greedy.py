#
# greedy.py
#
# This file provides a function implementing greedy best-first search for
# a route finding problem. Various search utilities from "route.py" are
# used in this function, including the classes RouteProblem, Node, and
# Frontier. Also, this function uses heuristic function objects defined
# in the "heuristic.py" file.
#
# YOUR COMMENTS INCLUDING CITATIONS
#
# YOUR NAME - THE DATE
#


from route import Node
from route import Frontier
import heuristic


def greedy_search(problem, h:heuristic.HeuristicFunction, repeat_check=False):
    """Perform greedy best-first search to solve the given route finding
    problem, returning a solution node in the search tree, corresponding
    to the goal location, if a solution is found. Only perform repeated
    state checking if the provided boolean argument is true."""

    # initialization
    startnode = Node(loc=problem.start,h_eval=h.h_cost(problem.start),h_fun=h)
    if problem.is_goal(startnode.loc):
        return startnode

    greedy_frontier = Frontier(startnode,sort_by='h')
    reached_set = {startnode}

    while not greedy_frontier.is_empty():
        # pop node from frontier
        current_node = greedy_frontier.pop()

        if problem.is_goal(current_node.loc):
            return current_node

        current_node_children = current_node.expand(problem)
        # add child nodes into frontier unless repeat_check is on and child is already in the reached set
        for child in current_node_children:
            if not repeat_check or child not in reached_set:
                greedy_frontier.add(child)
                reached_set.add(child)
    return None
