#
# dfs.py
#
# This file provides a function implementing depth-first search for a
# route-finding problem. Various search utilities from "route.py" are
# used in this function, including the classes RouteProblem, Node, and
# Frontier.
# 
# YOUR COMMENTS INCLUDING CITATIONS AND ACKNOWLEDGMENTS
#
# YOUR NAME - THE DATE
# 


from route import Node
from route import Frontier


def DFS(problem, repeat_check=False):
    """Perform depth-first search to solve the given route finding
    problem, returning a solution node in the search tree, corresponding
    to the goal location, if a solution is found. Only perform repeated
    state checking if the provided boolean argument is true."""

    # PLACE YOUR CODE HERE
    startnode = Node(loc=problem.start)
    if problem.is_goal(startnode.loc):
        return startnode

    bfs_frontier = Frontier(startnode, stack=True)
    reached_set = {startnode}

    while not bfs_frontier.is_empty():
        current_node = bfs_frontier.pop()

        if problem.is_goal(current_node.loc):
            return current_node

        current_node_children = current_node.expand(problem)

        for child in current_node_children:
            if not repeat_check or child not in reached_set:
                bfs_frontier.add(child)
                reached_set.add(child)

    return None