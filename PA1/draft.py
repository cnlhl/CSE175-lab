def uniform_cost_search(problem: RouteProblem, repeat_check=False):
    """Perform uniform cost search to solve the given route finding
    problem, returning a solution node in the search tree, corresponding
    to the goal location, if a solution is found. Only perform repeated
    state checking if the provided boolean argument is true."""

    startnode = Node(loc=problem.start)
    if problem.is_goal(startnode.loc):
        return startnode

    uniform_frontier = Frontier(startnode, sort_by='g')
    reached_set = {startnode.loc}  # Store locations, not Node objects

    while not uniform_frontier.is_empty():
        current_node = uniform_frontier.pop()

        if problem.is_goal(current_node.loc):
            return current_node

        current_node_children = current_node.expand(problem)

        for child in current_node_children:
            if repeat_check and child.loc in reached_set:  # Check by location
                if child in uniform_frontier:  # This uses the `__contains__` method
                    # The following checks if the path cost through this new child is shorter
                    # than the one already in the frontier.
                    if uniform_frontier[child] > child.path_cost:
                        # Delete the old node and insert the new one
                        del uniform_frontier[child]
                        uniform_frontier.add(child)
            else:
                uniform_frontier.add(child)
                reached_set.add(child.loc)  # Add the location to the reached set
    return None
