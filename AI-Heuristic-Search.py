import numpy as np
import copy
import math


class Node:
    '''
    This class represents a node object in an 8-puzzle problem.
    '''

    def __init__(self, parent=None, state=None, level=0, is_root=False, is_goal=False):
        '''
        Initiate a class state randomly if it is a root state.
        or
        As the requested goal state (by changing the values order in the list)
        or
        accept a a given state.
        '''
        self.is_goal = is_goal
        self.is_root = is_root
        self.level = level
        self.parent = parent

        if is_goal:
            self.state = np.array([1, 2, 3, 4, 5, 6, 7, 8, '_'], dtype=object)
            self.state = np.reshape(self.state, (3, 3))
        elif is_root:
            self.state = np.array([1, 2, 3, 4, 5, 6, 7, 8, '_'], dtype=object)
            np.random.shuffle(self.state)
            self.state = np.reshape(self.state, (3, 3))
        else:
            self.state = state

        if not is_goal:
            self.h = h(self)

    def get_position(self, num):
        '''
        This function returns a position of a value [row,column] indexes by given value.
        '''
        for row in range(len(self.state)):
            for col in range(len(self.state[row])):
                if self.state[row][col] == num:
                    return [row, col]

    def get_value(self, row, col):
        '''
        This function return a value in the puzzle by given indexes values.
        '''
        return self.state[row, col]

    def generate_child(self):
        '''
        This method generates sub-nodes of the node (move blank position p, down, right or left [if possible])
        and stores them in a list of the object.
        '''
        children = []
        blank = self.get_position('_')
        #                       left, right, down, up
        moves_list = np.array([[0, -1], [0, 1], [1, 0], [-1, 0]])
        for move in moves_list:
            swap_position = blank + move  # where the blank goes
            # check if movement is out of bound
            if ((swap_position[0] > 2) or
                    (swap_position[1] > 2) or
                    (swap_position[0] < 0) or
                    (swap_position[1] < 0)):
                continue
            swap_value = self.get_value(swap_position[0], swap_position[1])  # what was where the blank goes
            new_state = copy.deepcopy(self.state)  # duplicate state
            new_state[swap_position[0], swap_position[1]] = '_'  # put blank in new place
            new_state[blank[0], blank[1]] = swap_value  # swap value with blank
            if not self.is_root:
                if np.array_equal(new_state, self.parent.state):
                    continue
            children.append(Node(parent=self, state=new_state, level=self.level + 1))
        return children

    def __eq__(self, other):
        '''
        Node are compeared by its state simillarity
        '''
        return np.array_equal(self.state, other.state)


def is_solvable(node):
    """
    This boolean function check if a given node of the 8-puzzle problam can be solved.
    The answer is based on the even or odd number of inverses of tiles locations
    there are in the order of the given puzzle state.
    """
    order = []
    for row in range(len(node.state)):
        for col in range(len(node.state[row])):
            if node.state[row, col] == '_':
                continue
            order.append(node.state[row, col])  # Insert the order of current node state.
    inverses = []
    # For each value in the order
    # count how many numbers are greater then itself to the right
    for num, i in zip(order, range(len(order))):
        inverses.append(sum([j < num for j in order[i:]]))
    return sum(inverses) % 2 == 0


def print_solution(node):
    """This function sends a solution node to get the full solution path
        and then prints it, along with the solution's cost (length)"""
    steps = get_steps(node)  # Get the path
    for step in steps:
        # Print from start to finish
        print(step)
        print()
    print('cost:', len(steps) - 1)


def get_steps(node):
    """This function gets a node the retrieves the path from the root of the tree
        to that node, returning the path as a list of states"""
    if node.is_root:
        # If reached root, return its state
        return [node.state]
    else:
        # Add the current state to the previous states list
        return get_steps(node.parent) + [node.state]


def is_node_in_list(node, list_):
    '''
    This boolean function return whether a node (represented as matrix) is in a given list.
    '''
    for list_node in list_:
        if np.array_equal(node.state, list_node.state):
            return True
    return False


def initial_state():
    '''
    This function generates and returns an initial and solvable state of the 8-puzzle problem.
    '''
    have_solution = False
    while not have_solution:
        # Generate a root node until it's solvable
        start_node = Node(is_root=True)
        have_solution = is_solvable(start_node)
    return start_node

def manhattan_distance(node):
    """Heuristic for 8 puzzle: returns sum for each tile of manhattan
    distance between it's position in node's state and goal"""
    # Initialize the states
    sum = 0
    curr = node.state
    goal = Node(is_goal=True).state
    for num in [1, 2, 3, 4, 5, 6, 7, 8]:
        # For each number in the puzzle, find its location in the current and goal state
        i, j = np.where(curr == num)
        m, n = np.where(goal == num)

        # Add the absolute difference in x and y between the locations
        sum += abs(i[0] - m[0]) + abs(j[0] - n[0])

    return sum


def linear_conflict(node):
    '''
    This heuristic function returns the sum of linear conflicts in the 8-puzzle problem.
    The calculation is based on (the pairs in the correct row as the goal state but with reversed positions) * 2.
    '''

    # Set the initial states
    sum_conf = 0
    curr = node.state
    goal = Node(is_goal=True).state
    for r in range(3):
        # For each row form a list of all numbers in their correct row
        in_row = [num for num in goal[r].tolist() if (num in curr[r].tolist() and num != '_')]
        in_row.sort()
        if len(in_row) < 2:
            # No conflicts in the row, move on
            continue
        # For the other cases, calculate the cost of the linear conflicts
        elif len(in_row) == 2:
            i, j = np.where(curr == in_row[0])
            m, n = np.where(curr == in_row[1])
            if n < j:
                sum_conf += 2
        else:
            i, j = np.where(curr == in_row[0])
            m, n = np.where(curr == in_row[1])
            k, l = np.where(curr == in_row[2])
            if n < j:
                sum_conf += 2
            if l < n:
                sum_conf += 2

    return sum_conf


def solve_bnb(start=None):
    """This is a wrapper function for the BnB algorithm. It sets the initial
        variables and sends them to the BnB recursive function. The solution
        node is sent to another function to be printed"""
    if start is None:
        # If no start node is provided, generate it
        start = initial_state()
    if not is_solvable(start):
        # If the start node is unsolvable, don't run BnB
        return "Puzzle can not be solved"
    # Initialize the algorithm variables
    UB = math.inf
    goal = Node(is_goal=True)
    start.level = 0
    global nodes
    nodes = {}
    # Activate the algorithm and print the solution
    cost, solution = BnB([start], UB, goal, None)

    print_solution(solution)


def BnB(Q, UB, goal, solution):
    """This is a recursive function implementing the BnB algorithm. It attempts
        to reach from an initial state to the goal state in the optimal way,
        only looking at branches that are able to lead to a better solution than
        the existing one. Otherwise, the branches are pruned.

        Args:
        Q- the list of node to look at, only containing the latest nodes created
        UB- the cost of the best solution so far, starts as infinity
        goal- the goal node
        solution- the latest solution node found

        Returns:
        The cost of the solution
        The solution node
        """
    # global counter  # the counter is used the count node expansions
    # counter += 1

    global nodes  # The dictionary to hold all the created nodes, to avoid loops
    Q.sort(key=h)  # Sort the list of nodes by the heuristics

    for branch in Q:
        # For each node in the list set the lower bound as its level
        LB = branch.level

        # Create the node's key version and check it in the dictionary
        key = tuple(map(tuple, branch.state))
        if key not in nodes.keys():
            # If the node state is new, add it to the dictionary
            nodes[key] = branch
        elif LB < nodes[key].level:
            # If it's not new but is cheaper than the previous node, update it
            nodes[key] = branch
        else:
            # Duplicate node, skip it
            continue

        if LB + h(branch) >= UB:
            # If the estimated cost of the solution from this node exceeds the UB, skip it
            continue
        elif branch == goal:
            # If the goal is reached, update the solution node and UB
            solution = branch
            UB = LB
        else:
            # Expand this node and explore to find a solution, send to BnB
            next_steps = branch.generate_child()
            UB, solution = BnB(next_steps, UB, goal, solution)

    return UB, solution


def f(node):
    '''
    This function returns the sort key for the A star algorithm,
    by getting a node and calculating it's depth and heuristic measure
    to the goal state by built-in heuristic call to function called 'h'.
    '''
    return node.level + node.h


def A_star(start=None):
    '''
    This function performes a heuristic search using A star algorithm.
    The algorithm finds the optimal path from the start state to goal state
    by holding a "open list" and a "close list", where he first ist used to store
    all the nodes to be visited and the second one to store the nodes that has been visited
    and expanded.

    The "open list" is sorted by a specific function that i based of node state.
    The function uses a built-in heuristic call for open list sort function called 'f'.
    The algorithm "pulls" every time the first node in the "open list" and visits it
    so it checks whether it is the goal state node, otherwise, it should expand this node and keep searching.

    The function can receive a start node or initialize one by itself.
    '''
    open_list = []
    closed_list = []
    if start is None:
        start = initial_state()
    if not is_solvable(start):
        # If the start node is unsolvable, don't run BnB
        return "Puzzle can not be solved"
    open_list.append(start)
    while True:
        current_node = open_list[0]
        if h(current_node) == 0:
            print_solution(current_node)
            break
        else:
            children = current_node.generate_child()
            for child in children:
                if is_node_in_list(child, closed_list):
                    continue
                open_list.append(child)
        closed_list.append(current_node)
        del open_list[0]
        open_list.sort(key=f, reverse=False)


def h(node):
    """This function receives a node and returns the heuristic value for the
        node's distance from the solution. It can be adjusted to suit any
        heuristic function that was implemented"""

    # return manhattan_distance(node) + linear_conflict(node)
    return manhattan_distance(node)


# Main:

def main():

    # For changing the heuristic please only edit a single line in the "h" function above.

    start_node = initial_state()
    solve_bnb(start_node)
    A_star(start_node)


if __name__ == '__main__':
    main()
