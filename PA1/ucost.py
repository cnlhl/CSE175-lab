#
# ucost.py
#
# This file provides a function implementing uniform cost search for a
# route finding problem. Various search utilities from "route.py" are
# used in this function, including the classes RouteProblem, Node, and
# Frontier.
#
# YOUR COMMENTS INCLUDING CITATIONS
#
# YOUR NAME - THE DATE
#


from route import Node
from route import Frontier
from route import RouteProblem


def uniform_cost_search(problem:RouteProblem, repeat_check=False):
    """Perform uniform cost search to solve the given route finding
    problem, returning a solution node in the search tree, corresponding
    to the goal location, if a solution is found. Only perform repeated
    state checking if the provided boolean argument is true."""

    # initialization
    startnode = Node(loc=problem.start)
    if problem.is_goal(startnode.loc):
        return startnode

    # sort by path cost(g value)
    uniform_frontier = Frontier(startnode,sort_by='g')
    reached_set = {startnode}

    while not uniform_frontier.is_empty():
        current_node = uniform_frontier.pop()

        if problem.is_goal(current_node.loc):
            return current_node

        current_node_children = current_node.expand(problem)
        # if repeat_check is on, check repeat status and whether to replace existed node;
        for child in current_node_children:
            if repeat_check and child in reached_set:
                if uniform_frontier.contains(child):
                    if uniform_frontier[child] > child.path_cost:
                        del uniform_frontier[child]
                        uniform_frontier.add(child)
            # otherwise, directly add the child nodes
            else:
                uniform_frontier.add(child)
                reached_set.add(child)
    return None


