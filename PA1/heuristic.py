#
# heuristic.py
#
# This script defines a utility class that can be used as an implementation
# of a frontier state (location) evaluation function for use in route-finding
# heuristic search algorithms. When a HeuristicSearch object is created,
# initialization code can be executed to prepare for the use of the heuristic
# during search. In particular, a RouteProblem object is typically provided 
# when the HeuristicFunction is created, providing information potentially
# useful for initialization. The actual heuristic cost function, simply
# called "h_cost", takes a state (location) as an argument.
#
# YOUR COMMENTS INCLUDING CITATIONS
#
# YOUR NAME - THE DATE
#


import route


class HeuristicFunction:
    """A heuristic function object contains the information needed to
    evaluate a state (location) in terms of its proximity to an optimal
    goal state."""

    def __init__(self, problem:route.RouteProblem=None):
        self.problem = problem
        # PLACE ANY INITIALIZATION CODE HERE
        self.min_consume = self.minimum_cost_per_distance(problem.map)

    def minimum_cost_per_distance(self,road_map:route.RoadMap):
        min_value = float("inf")  # Start with infinity so any actual value is smaller

        # Iterating over each location and its connections
        for start_loc, connections in road_map.connection_dict.items():
            for end_loc, cost in connections.items():
                # Calculating euclidean distance for the current road segment
                distance = road_map.euclidean_distance(start_loc, end_loc)
                if distance == 0:  # Avoiding division by zero
                    continue

                value = cost / distance
                if value < min_value:
                    min_value = value

        return min_value

    def h_cost(self, loc=None):
        """An admissible heuristic function, estimating the cost from
        the specified location to the goal state of the problem."""
        # a heuristic value of zero is admissible but not informative
        value = 0.0
        if loc is None:
            return value
        else:
            # PLACE YOUR CODE FOR CALCULATING value OF loc HERE
            eu_distance = self.problem.map.euclidean_distance(loc,self.problem.goal)
            value = eu_distance * self.min_consume
            return value

