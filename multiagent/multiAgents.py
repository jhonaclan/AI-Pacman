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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
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
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        if currentGameState.isLose():
            return -9999.0
        elif currentGameState.isWin():
            return 9999.0

        # Helper function to get minimum distance from a list of positions
        def getMinDistance(positions, fromPos):
            return min([manhattanDistance(pos, fromPos) for pos in positions]) if positions else 0

        # Award and penalty initialization
        award_score = 0.0
        penalty_score = 0.0

        foods = newFood.asList()
        capsules = successorGameState.getCapsules()

        # Calculate food and capsule distances
        closest_food_distance = getMinDistance(foods, newPos)
        closest_capsule_distance = getMinDistance(capsules, newPos)

        # Prioritize capsules over food
        award_score += 1.0 / (closest_food_distance + 1)
        award_score += 5.0 / (closest_capsule_distance + 1)

        # Ghost proximity considerations
        ghost_positions = [ghostState.getPosition() for ghostState in newGhostStates]
        closest_ghost_distance = getMinDistance(ghost_positions, newPos)
        if closest_ghost_distance <= 2:
            penalty_score += (3 - closest_ghost_distance) * 100  # Increase penalty the closer the ghost is

        # Do not stop unless necessary
        if action == 'Stop':
            penalty_score += 10

        return successorGameState.getScore() + sum(newScaredTimes) + award_score - penalty_score

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
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
        def minValue(state, agentIndex, depth):
            # Check for terminal state
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            # Get legal actions and initialize value
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(state)

            v = float('inf')
            for action in legalActions:
                if agentIndex == gameState.getNumAgents() - 1:  # if last ghost
                    v = min(v, maxValue(state.generateSuccessor(agentIndex, action), depth+1))
                else:  # if not the last ghost
                    v = min(v, minValue(state.generateSuccessor(agentIndex, action), agentIndex+1, depth))
            return v

        def maxValue(state, depth):
            # Check for terminal state
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            # Get legal actions and initialize value
            legalActions = state.getLegalActions(0)
            if not legalActions:
                return self.evaluationFunction(state)

            v = float('-inf')
            for action in legalActions:
                v = max(v, minValue(state.generateSuccessor(0, action), 1, depth))
            return v

        # Root node: Get best action for Pacman
        actions = gameState.getLegalActions(0)
        bestAction = max(actions, key=lambda action: minValue(gameState.generateSuccessor(0, action), 1, 0))
        return bestAction

        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def minValue(state, agentIndex, depth, alpha, beta):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            v = float('inf')
            legalActions = state.getLegalActions(agentIndex)

            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                if agentIndex == gameState.getNumAgents() - 1:  # If this is the last ghost
                    v = min(v, maxValue(successor, depth + 1, alpha, beta))
                else:
                    v = min(v, minValue(successor, agentIndex + 1, depth, alpha, beta))
                if v < alpha:  # Prune
                    return v
                beta = min(beta, v)
            return v

        def maxValue(state, depth, alpha, beta):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            v = float('-inf')
            legalActions = state.getLegalActions(0)

            for action in legalActions:
                successor = state.generateSuccessor(0, action)
                v = max(v, minValue(successor, 1, depth, alpha, beta))
                if v > beta:  # Prune
                    return v
                alpha = max(alpha, v)
            return v

        # Driver code starts here:
        alpha = float('-inf')
        beta = float('inf')
        bestValue = float('-inf')
        bestAction = None

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = minValue(successor, 1, 0, alpha, beta)
            if value > bestValue:
                bestValue = value
                bestAction = action
            alpha = max(alpha, bestValue)

        return bestAction

        util.raiseNotDefined()

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

        def expectimax(state, agentIndex, depth):
            """
            The expectimax function calculates the expected value of the game state
            for the agent at the given index and search depth.
            """
            # Base case: return the evaluation function value
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            legalActions = state.getLegalActions(agentIndex)

            # Expected value calculation for ghosts
            if agentIndex != 0:  # if it's a ghost
                expectedValue = sum(
                    expectimax(state.generateSuccessor(agentIndex, action),
                               (agentIndex + 1) % state.getNumAgents(),
                               depth + (agentIndex + 1) // state.getNumAgents())
                    for action in legalActions) / len(legalActions)
                return expectedValue

            # Max value calculation for Pacman
            else:
                return max(
                    expectimax(state.generateSuccessor(agentIndex, action),
                               (agentIndex + 1) % state.getNumAgents(), depth)
                    for action in legalActions)

        # Choose the action that maximizes the expected utility
        return max(gameState.getLegalActions(0),
                   key=lambda action: expectimax(gameState.generateSuccessor(0, action), 1, 0))

        util.raiseNotDefined()

        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Improved evaluation function with weighted scores for various
    aspects of the game like food distance, capsule distance, and ghost distance.

    This function computes an evaluation score for Pacman's current state by considering several crucial factors:

    1. **Game Status**: Awards a high positive score for winning states and a large negative score for losing states.
    2. **Food Proximity**: Encourages Pacman to stay closer to the nearest food pellet. The closer the food, the higher the score.
    3. **Capsule Proximity**: Prioritizes staying closer to power capsules, as they can turn ghosts vulnerable.
    4. **Ghost Interaction**:
        - When ghosts are vulnerable (scared): The function encourages Pacman to chase the ghosts. However, if the scare timer is nearing its end, Pacman is discouraged from pursuing distant ghosts.
        - When ghosts are not scared: Keeps Pacman at a safe distance to avoid getting caught.

    Overall, the function aims to maximize Pacman's score by eating food and capsules while safely navigating around the ghosts.
    """
    if currentGameState.isWin():
        return 99999

    if currentGameState.isLose():
        return -99999

    currentPos = currentGameState.getPacmanPosition()
    currentFood = currentGameState.getFood().asList()
    currentGhostStates = currentGameState.getGhostStates()
    currentScaredTimes = [ghostState.scaredTimer for ghostState in currentGhostStates]
    currentCapsule = currentGameState.getCapsules()

    score = 0

    foodDistance = [util.manhattanDistance(currentPos, food) for food in currentFood]
    if foodDistance:
        nearestFood = min(foodDistance)
        score += 1.0 / max(nearestFood, 1)

    score -= len(currentFood)

    if currentCapsule:
        capsuleDistance = [util.manhattanDistance(currentPos, capsule) for capsule in currentCapsule]
        nearestCapsule = min(capsuleDistance)
        score += 2.0 / max(nearestCapsule, 1)

    currentGhostDistances = [util.manhattanDistance(currentPos, ghost.getPosition()) for ghost in currentGhostStates]
    nearestCurrentGhost = min(currentGhostDistances)
    scaredTime = sum(currentScaredTimes)

    if scaredTime > 0:
        if nearestCurrentGhost <= scaredTime:
            score += 2.0 / max(nearestCurrentGhost, 1)
    else:
        if nearestCurrentGhost <= 1:
            score -= 4.0 / max(nearestCurrentGhost, 1)

    return currentGameState.getScore() + score

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
