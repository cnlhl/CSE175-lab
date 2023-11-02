#
# astar.py
#
# This file provides a function implementing A* search for a route finding
# problem. Various search utilities from "route.py" are used in this function,
# including the classes RouteProblem, Node, and Frontier. Also, this function
# uses heuristic function objects defined in the "heuristic.py" file.
#
# YOUR COMMENTS INCLUDING CITATIONS
#
# YOUR NAME - THE DATE
#


from route import Node
from route import Frontier


def a_star_search(problem, h, repeat_check=False):
    """Perform A-Star search to solve the given route finding problem,
    returning a solution node in the search tree, corresponding to the goal
    location, if a solution is found. Only perform repeated state checking if
    the provided boolean argument is true."""

    # initialization
    startnode = Node(loc=problem.start,h_eval=h.h_cost(problem.start),h_fun=h)
    if problem.is_goal(startnode.loc):
        return startnode
    # sort by f value: (h+g)
    astar_frontier = Frontier(startnode,sort_by='f')
    reached_set = {startnode}

    while not astar_frontier.is_empty():
        current_node = astar_frontier.pop()

        if problem.is_goal(current_node.loc):
            return current_node

        current_node_children = current_node.expand(problem)
        # if repeat check is on, compare exited node's value to decide whether to replace
        for child in current_node_children:
            if repeat_check and child in reached_set:
                if astar_frontier.contains(child):
                    if astar_frontier[child] > child.value('f'):
                        del astar_frontier[child]
                        astar_frontier.add(child)
            # otherwise directly add the child nodes
            else:
                astar_frontier.add(child)
                reached_set.add(child)
    return None
