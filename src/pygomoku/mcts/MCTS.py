import abc
import copy

import numpy as np
import six

from pygomoku.Board import Board
from pygomoku.mcts import policy_fn
from pygomoku.mcts.progressbar import ProgressBar


def softmax(x):
    probs = np.exp(x - np.max(x))  # avoid overflow
    probs /= np.sum(probs)
    return probs


def action_prob_via_vis_times(vis_times):
    activates = vis_times - np.max(vis_times)
    probs = activates / np.sum(activates)
    return probs


@six.add_metaclass(abc.ABCMeta)
class TreeNode(object):
    """
    The abstract class for tree node in search tree.
    """
    @abc.abstractmethod
    def expand(self, action_priors):
        """Expand tree node by creating new child.

        Args:
            action_priors:
                a list of avaliable actions(for UCT).
                a list of tuple of avaliable actions and their prior probs.          
        """
        pass

    @abc.abstractmethod
    def select(self, weight_c):
        """Select action among children of current node.

        Return: 
            A tuple, (action, next_node)
        """
        pass

    @abc.abstractmethod
    def update(self, bp_value):
        """Update node value from leaf.
        """
        pass

    @abc.abstractmethod
    def backPropagation(self, bp_value):
        """Backpropagation the final result from leaf to the root.
        """
        pass

    @abc.abstractmethod
    def evaluate(self, weight_c):
        """Calculate and return the UCB value for this node.

        Args:
            weight_c: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        pass

    @abc.abstractmethod
    def is_root(self):
        pass

    @abc.abstractmethod
    def is_leaf(self):
        pass


@six.add_metaclass(abc.ABCMeta)
class TreeSearch(object):
    """The abstract class for tree search.
    """

    @abc.abstractmethod
    def _playout(self, state):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        pass

    # @abc.abstractmethod
    # def _evaluateRollout(self, state, limit):
    #     """Use the rollout policy to play until the end of the game,
    #     returning +1 if the current player wins, -1 if the opponent wins,
    #     and 0 if it is a tie.
    #     """
    #     pass

    @abc.abstractmethod
    def getMove(self, state):
        """Runs all playouts sequentially and returns the most visited action.
        state: the current game state

        Return: the selected action
        """
        pass

    @abc.abstractmethod
    def updateWithMove(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        pass
    
    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def __str__(self):
        pass


class MCTSTreeNode(TreeNode):
    """A node in the MCTS tree. Each node keeps track of its own value Q,
    prior probability P, and its visit-count-adjusted prior score u.

    Attributes:
        parent: The parent node for current node. Root's parent is None.
        children: A dict whose key is action and value is corresponding child node.
        _vis_times: An integer shows the number of times this node has been visited.
        _Q: Q value, the quality value. Judge the value for exploitation for a node.
        _U: U value. Judge the value for exploration for a node. A node with more 
            visit times will have small U value.
        _P: The prior probability for a node to be exploration(or the 
            prior probability for its corresponding action to be taken).
    """

    def __init__(self, parent, prior_prob):
        self.parent = parent
        self.children = {}  # a map from action to node
        self._vis_times = 0
        self._Q = 0  # Q = sum_{all rollout}(rollout_result)/vis_times
        self._U = 0  # U = prior_prob / (1 + vis_times)
        self._P = prior_prob

    def expand(self, action_priors):
        """Expand this node by creating all its children.

        Args:
            action_priors: the (action, prior probability) list for its children node.
        """
        for action, prob in action_priors:
            if action not in self.children:
                self.children[action] = MCTSTreeNode(self, prob)

    def select(self, weight_c):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).

        Return: A tuple of (action, next_node)
        """
        return max(self.children.items(),
                   key=lambda act_node: act_node[1].evaluate(weight_c))

    def update(self, bp_value):
        """Update node values from leaf evaluation.

        vis_time += 1
        Q += delta

        Args:
            bp_value: the value of subtree evaluation from the current player's
                perspective.
        """
        self._vis_times += 1

        # this expression of Q is a running average
        # suppose v_{i} is the result value of i-th rollout
        # Q_{N} = sum_{1 to N}(v) / N
        # Q_{N+1}
        #   = (sum_{1 to N}(v) + v_{N+1}) / (N+1)
        #   = N/(N+1) * Q_{N} + v_{N+1}/(N+1)
        # dQ
        #   = Q_{N+1} - Q_{N}
        #   = (v_{N+1} - Q_{N}) / (N+1)

        self._Q += float(bp_value - self._Q) / self._vis_times

    def backPropagation(self, bp_value):
        """Backpropagation the final result from leaf to the root.
        """
        self.update(bp_value)
        if self.parent:  # if has parent, not root
            # NOTE: '-' --> good result for one player means bad result for the other.
            self.parent.backPropagation(-bp_value)

    def evaluate(self, weight_c):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.

        Args:
            weight_c: a number in (0, inf) controlling the relative impact of
                value Q, and prior probability P, on this node's score.
        """
        self._U = self._P * \
            np.sqrt(self.parent._vis_times) / (1 + self._vis_times)
        return self._Q + weight_c * self._U

    def is_leaf(self):
        return False if self.children else True

    def is_root(self):
        return self.parent is None

    @property
    def vis_times(self):
        return self._vis_times

    @property
    def Q_value(self):
        return self._Q


class MCTS(TreeSearch):
    """
    The Monte Carlo Tree Search.

    Attributes:
        root: The root node for search tree.
        _expand_policy: A function that takes in a board state and outputs
            a list of (action, probability) tuples which used for node expanding 
            and also a score between in [-1,1] (i.e. The expected value of the end 
            game score from the current player's perspective, in pure MCTS without
            Neural network, this value will be 0) for the current player.
        _rollout_policy: A function similar to expand_policy, used for random play
            in rollout phase.
        _weight_c: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior less and prefer to try new node.
        _compute_budget: How many times will we search in this tree (Num of playout).
        _silent: If True, MCTS will not print log informations.
        _expand_bound: Only expand a leaf node when its vis_times >= expand_bound
    """

    def __init__(self, expand_policy, rollout_policy, weight_c=5, compute_budget=10000, expand_bound=1, silent=False):
        self.root = MCTSTreeNode(None, 1.0)
        self._expand_policy = expand_policy
        self._rollout_policy = rollout_policy
        self._weight_c = weight_c
        self._compute_budget = int(compute_budget)
        self._silent = silent
        self._expand_bound = min(expand_bound, compute_budget)
    
    def reset(self):
        self.root = MCTSTreeNode(None, 1.0)

    def _playout(self, state):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self.root
        while True:
            if node.is_leaf():  # if leaf or only root in tree.
                break

            action, node = node.select(self._weight_c)
            state.play(action)

        action_probs, _ = self._expand_policy(state)
        # Check for end of game
        is_end, _ = state.gameEnd()
        if not is_end and node.vis_times >= self._expand_bound:
            node.expand(action_probs)

        # Evaluate the leaf node by random rollout
        bp_value = self._evaluateRollout(state)
        # bp
        node.backPropagation(-bp_value)

    def _evaluateRollout(self, state, limit=1000):
        """Use the rollout policy to play until the end of the game,
        returning +1 if the current player wins, -1 if the opponent wins,
        and 0 if it is a tie.

        Args:
            state: current board state
            limit: usually in gomoku we don't need this. The upper bound for 
                rollout times.
        """
        # player color of the leaf node
        player_color = state.current_player

        for _ in range(limit):
            is_end, winner_color = state.gameEnd()
            if is_end:
                break
            action_probs = self._rollout_policy(state)  # (action, prob)
            next_action = max(action_probs, key=lambda x: x[1])[0]
            state.play(next_action)
        else:
            if not self._silent:
                print("[Warning]: rollout exceeds the limit({})".format(limit))

        if winner_color is None:
            return 0
        else:
            return 1 if winner_color == player_color else -1

    def getMove(self, state):
        """Runs all playouts sequentially and returns the most visited action.
        state: the current game state

        Return: the selected action
        """
        # if at the beginning of game, we should put stone at center.
        if state.is_empty:
            return len(state.availables) // 2

        if self._silent:
            for _ in range(self._compute_budget):
                state_copy = copy.deepcopy(state)
                self._playout(state_copy)
        else:
            print("Thinking...")
            pb = ProgressBar(self._compute_budget, total_sharp=20)
            for _ in range(self._compute_budget):
                pb.iterStart()
                state_copy = copy.deepcopy(state)
                self._playout(state_copy)
                pb.iterEnd()

        return max(self.root.children.items(),
                   key=lambda act_node: act_node[1].vis_times)[0]
        # return max(self.root.children.items(),
        #            key=lambda act_node: act_node[1].Q_value)[0]
    
    def testOut(self):
        return sorted(list(self.root.children.items()), key=lambda x: x[-1])

    def think(self, state, decay_level=100):
        """Consider the current board state and give a suggested move.

        Similar to getMove but with less compute budget. Usually used to
        gauss the next move of opponent or give tips to human player.

        Args:
            state: Current board state.
            decay_level: A value describe the importence of this think action. 
                A higher value means MCTS will pay less attention to this 'think action'.
        """
        for _ in range(self._compute_budget//decay_level):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
        return max(self.root.children.items(),
                   key=lambda act_node: act_node[1].vis_times)[0]

    def updateWithMove(self, last_move):
        """Reuse the Tree, and take a step forward.
        """
        if last_move in self.root.children:  # if can reuse
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:   # else rebuild the tree
            self.root = MCTSTreeNode(None, 1.0)

    def __str__(self):
        return "MCTS with compute budget {} and weight c {}".format(self._compute_budget, self._weight_c)

    @property
    def silent(self):
        return self._silent
    @silent.setter
    def silent(self, given_value):
        if isinstance(given_value, bool):
            self._silent = given_value


class MCTSWithDNN(TreeSearch):
    """
    The Monte Carlo Tree Search using deep neural network.

    Attributes:
        root: The root node for search tree.
        _policy_value_fn: A function that takes in a board state and outputs
            a list of (action, probability) tuples which used for node expanding 
            and also a score between in [-1,1] (i.e. The expected value of the end 
            game score from the current player's perspective, in pure MCTS without
            Neural network, this value will be 0) for the current player.
        _weight_c: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more and prefer to try new node.
        _compute_budget: How many times will we search in this tree (Num of playout).
        _silent: If True, MCTS will not print log informations.
        _expand_bound: Only expand a leaf node when its vis_times >= expand_bound
    """

    def __init__(self, policy_value_fn, weight_c=5, compute_budget=10000,
                 expand_bound=10, silent=False):
        self.root = MCTSTreeNode(None, 1.0)
        self._policy_value_fn = policy_value_fn
        self._weight_c = weight_c
        self._compute_budget = int(compute_budget)
        self._silent = silent
        self._expand_bound = min(expand_bound, compute_budget)

    def _playout(self, state):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back throuht the path from the leaf to the root.
        State is modified in-place, so a copy must be provided.
        """
        node = self.root
        while True:
            if node.is_leaf():  # if leaf or only root
                break
            action, node = node.select(self._weight_c)
            state.play(action)

        # Here DNN out value will replace rollout value
        policy, value = self._policy_value_fn(state)
        # Check for end of game
        is_end, winner = state.gameEnd()
        if not is_end: 
            if node.vis_times >= self._expand_bound:
                node.expand(policy)
        else:
            if winner is None:
                value = 0.0
            else:
                value = 1.0 if state.current_player == winner else -1.0

        # back propagation
        node.backPropagation(-value)

    def getMove(self, state, exploration_level):
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.

        Args:
            state: Current board state.
            exploration_level: temperature parameter in (0, 1] controls 
                the level of exploration.

        Return:
            All vaild actions with their probabilties.
        """
        if self._silent:
            for _ in range(self._compute_budget):
                state_copy = copy.deepcopy(state)
                self._playout(state_copy)
        else:
            print("Thinking...")
            pb = ProgressBar(self._compute_budget)
            for _ in range(self._compute_budget):
                pb.iterStart()
                state_copy = copy.deepcopy(state)
                self._playout(state_copy)
                pb.iterEnd()

        # calculate the move probabilities based on visit
        # counts at the root node
        act_vis = [(act, node.vis_times)
                   for act, node in self.root.children.items()]
        acts, visits = zip(*act_vis)

        # Softmax version
        # Can use temperature param to do some smooth
        probs = softmax(1.0/exploration_level * np.log(np.array(visits) + 1e-10))

        # Naive version
        # This one will suffer from some numerical problems.
        # probs = action_prob_via_vis_times(visits)

        return acts, probs

    def think(self, state, decay_level=100):
        """Consider the current board state and give a suggested move.

        Similar to getMove but with less compute budget. Usually used to
        gauss the next move of opponent or give tips to human player.

        Args:
            state: Current board state.
            decay_level: A value describe the importence of this think action. 
                A higher value means MCTS will pay less attention to this 'think action'.
        """
        for _ in range(self._compute_budget // decay_level):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
        act_vis = [(act, node.vis_times)
                   for act, node in self.root.children.items()]
        act, visits = zip(*act_vis)
        probs = action_prob_via_vis_times(visits)
        return act, probs


    def updateWithMove(self, last_move):
        """Reuse the Tree, and take a step forward.
        """
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            self.root = MCTSTreeNode(None, 1.0)
    
    def reset(self):
        self.root = MCTSTreeNode(None, 1.0)

    def __str__(self):
        return "MCTS(DNN version) with compute budget {} and weight c {}".format(self._compute_budget, self._weight_c)
    
    @property
    def silent(self):
        return self._silent
    @silent.setter
    def silent(self, given_value):
        if isinstance(given_value, bool):
            self._silent = given_value
            
    __repr__ = __str__
