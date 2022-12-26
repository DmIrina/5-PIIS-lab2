# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

import util
from util import manhattanDistance
from game import Directions
from game import Agent
import random
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legal_moves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = random.choice(best_indices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successor_game_state = currentGameState.generatePacmanSuccessor(action)
        new_pos = successor_game_state.getPacmanPosition()
        new_food = successor_game_state.getFood()

        "*** YOUR CODE HERE ***"
        score = successor_game_state.getScore()
        all_foods = new_food.asList()
        closest_ghost = min([manhattanDistance(currentGameState.getPacmanPosition(), ghostCoord.getPosition())
                             for ghostCoord in currentGameState.getGhostStates()])

        capsules = currentGameState.getCapsules()
        shortest_distance = 1000

        for food in all_foods:
            distance = manhattanDistance(food, new_pos)
            if distance < shortest_distance:
                shortest_distance = distance

        score += max(closest_ghost, 3)

        if len(all_foods) < len(currentGameState.getFood().asList()):
            score += 100

        score += 100 / shortest_distance

        if new_pos in capsules:
            score += 200

        if action == Directions.STOP:
            score -= 10

        return score


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.minimax(gameState, 0, self.depth)[1]

    def minimax(self, gameState, agent_index, depth):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState), "Stop"

        agents_num = gameState.getNumAgents()
        agent_index %= agents_num

        if agent_index == agents_num - 1:
            depth -= 1

        if agent_index == 0:
            return self.max_value(gameState, agent_index, depth)

        else:
            return self.min_value(gameState, agent_index, depth)

    def max_value(self, game_state, agent_index, depth):
        actions = []
        legal_actions = game_state.getLegalActions(agent_index)

        for action in legal_actions:
            actions.append(
                (self.minimax(game_state.generateSuccessor(agent_index, action), agent_index + 1, depth)[0], action))
        return max(actions)

    def min_value(self, game_state, agent_index, depth):
        actions = []
        legal_actions = game_state.getLegalActions(agent_index)

        for action in legal_actions:
            actions.append(
                (self.minimax(game_state.generateSuccessor(agent_index, action), agent_index + 1, depth)[0], action))
        return min(actions)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.max_value(gameState, 0, 0, -float("inf"), float("inf"))[0]

    def alpha_beta_decision(self, game_state, agent_index, depth, alpha, beta):
        agents_num = game_state.getNumAgents()

        if depth is self.depth * agents_num or game_state.isLose() or game_state.isWin():
            return self.evaluationFunction(game_state)

        if agent_index is 0:
            return self.max_value(game_state, agent_index, depth, alpha, beta)[1]

        else:
            return self.min_value(game_state, agent_index, depth, alpha, beta)[1]

    def max_value(self, game_state, agent_index, depth, alpha, beta):
        legal_actions = game_state.getLegalActions(agent_index)
        agents_num = game_state.getNumAgents()
        best_action = ("max", -float("inf"))

        for action in legal_actions:
            successor_action = (action, self.alpha_beta_decision(game_state.generateSuccessor(agent_index, action),
                                                                 (depth + 1) % agents_num,
                                                                 depth + 1,
                                                                 alpha,
                                                                 beta))

            best_action = max(best_action, successor_action, key=lambda x: x[1])

            if best_action[1] > beta:
                return best_action
            else:
                alpha = max(alpha, best_action[1])

        return best_action

    def min_value(self, game_state, agent_index, depth, alpha, beta):
        legal_actions = game_state.getLegalActions(agent_index)
        agents_num = game_state.getNumAgents()
        best_action = ("min", float("inf"))

        for action in legal_actions:
            successor_action = (action, self.alpha_beta_decision(game_state.generateSuccessor(agent_index, action),
                                                                 (depth + 1) % agents_num,
                                                                 depth + 1,
                                                                 alpha,
                                                                 beta))
            best_action = min(best_action, successor_action, key=lambda x: x[1])

            if best_action[1] < alpha:
                return best_action
            else:
                beta = min(beta, best_action[1])

        return best_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        agents_num = gameState.getNumAgents()
        max_depth = self.depth * agents_num
        return self.expectimax(gameState, "expect", max_depth, 0)[0]

    def expectimax(self, game_state, action, depth, agent_index):

        if depth is 0 or game_state.isLose() or game_state.isWin():
            return action, self.evaluationFunction(game_state)

        # if pacman (max agent) - return max successor value
        if agent_index is 0:
            return self.max_value(game_state, action, depth, agent_index)
        # if ghost (exp agent) - return probability value
        else:
            return self.exp_value(game_state, action, depth, agent_index)

    def max_value(self, game_state, action, depth, agent_index):
        legal_actions = game_state.getLegalActions(agent_index)
        agents_num = game_state.getNumAgents()
        best_action = ("max", -(float('inf')))

        for legal_action in legal_actions:
            next_agent = (agent_index + 1) % agents_num
            successor_action = None

            if depth != self.depth * game_state.getNumAgents():
                successor_action = action

            else:
                successor_action = legal_action

            successor_value = self.expectimax(game_state.generateSuccessor(agent_index, legal_action),
                                              successor_action, depth - 1, next_agent)

            best_action = max(best_action, successor_value, key=lambda x: x[1])

        return best_action

    def exp_value(self, game_state, action, depth, agent_index):
        legal_actions = game_state.getLegalActions(agent_index)
        average_score = 0
        probability = 1.0 / len(legal_actions)

        for legal_action in legal_actions:
            next_agent = (agent_index + 1) % game_state.getNumAgents()
            best_action = self.expectimax(game_state.generateSuccessor(agent_index, legal_action),
                                          action, depth - 1, next_agent)
            average_score += best_action[1] * probability

        return action, average_score


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    new_pos = currentGameState.getPacmanPosition()
    new_food = currentGameState.getFood().asList()
    min_food_list = float('inf')

    for food in new_food:
        min_food_list = min(min_food_list, manhattanDistance(new_pos, food))

    ghost_distance = 0

    for ghost in currentGameState.getGhostPositions():
        ghost_distance = manhattanDistance(new_pos, ghost)
        if ghost_distance < 2:
            return -float('inf')

    food_left = currentGameState.getNumFood()
    capsule_left = len(currentGameState.getCapsules())

    food_left_multiplier = 800000
    capsule_left_multiplier = 8500
    food_distance_multiplier = 800
    additional_factors = 0

    if currentGameState.isLose():
        additional_factors -= 40000
    elif currentGameState.isWin():
        additional_factors += 40000

    return (1.0 / (food_left + 1) * food_left_multiplier + ghost_distance) + \
           (1.0 / (min_food_list + 1) * food_distance_multiplier) + \
           (1.0 / (capsule_left + 1) * capsule_left_multiplier + additional_factors)


# Abbreviation
better = betterEvaluationFunction
