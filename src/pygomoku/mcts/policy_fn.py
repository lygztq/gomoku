import numpy as np


def rollout_policy_fn(board):
    """A random and fast version of policy function 
    used in rollout phase(also called default policy).

    Rollout phase(default policy) means we play randomly 
    until the game terminated and get the final result. We 
    then back-propagate this result to all nodes from the
    terminal state node to the root node.

    This method gives a random probability of each action corresponding
    to the currrent board state, which help us to play a step randomly.

    Args:
        board: current (leaf node if use MCTS) state in rollout phase.

    Return:
        a list with format [e1,e2,...](e=[action, probability])
    """
    action_probs = np.random.rand(len(board.availables))
    return zip(board.availables, action_probs)


def MCTS_expand_policy_fn(board):
    """A evaluate function used in expand phase in MCTS.

    This function takes a board state and return the 
    avaliable action combined with its prior probability(all the same)
    to expand current node in MCT. The function also return the evaluation 
    value for current board state(which is 0).

    Args:
        board: current (leaf node if use MCTS) state.

    Return:
        a list with format [e1,e2,...](e=[action, prob]) and a value.
    """
    action_probs = np.ones(len(board.availables)) / len(board.availables)
    return zip(board.availables, action_probs), 0
