import java.util.Arrays;
import java.util.Deque;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.ArrayList;

import aiinterface.CommandCenter;
import enumerate.Action;
import simulator.Simulator;
import struct.CharacterData;
import struct.FrameData;
import struct.GameData;
import struct.MotionData;

/**
 * Represents a node in the Monte Carlo Tree Search (MCTS) algorithm.
 * This implementation uses Upper Confidence Bound UCB1 for tree traversal
 * and handles action selection for fighting game AI.
 */
public class MCTSNode {

    /** Maximum time allocated for UCT search in nanoseconds */
    public static final int UCT_MAX_TIME = 165 * 100000;
    
    /** Maximum number of iterations for UCT search */
    public static final int UCT_ITERATION_LIMIT = 23;

    /** Exploration constant for UCB1 formula */
    public static final double UCB1_EXPLORATION_CONSTANT = 3;

    /** Maximum depth for tree expansion */
    public static final int UCT_MAX_DEPTH = 2;

    /** Minimum visit count required to expand a node */
    public static final int UCT_EXPANSION_THRESHOLD = 10;

    /** Duration for simulation/rollout in frames */
    public static final int SIMULATION_DURATION = 60;

    /** Random number generator for exploration */
    private Random randomGenerator;

    /** Parent node in the search tree */
    private MCTSNode parentNode;

    /** Child nodes in the search tree */
    private MCTSNode[] childNodes;

    /** Depth of this node in the search tree */
    private int nodeDepth;

    /** Number of times this node has been visited */
    private int visitCount;

    /** UCB1 value of this node */
    private double ucbValue;

    /** Accumulated score from simulations */
    private double accumulatedScore;

    /** Available actions for the player */
    private LinkedList<Action> availablePlayerActions;

    /** Available actions for the opponent */
    private LinkedList<Action> availableOpponentActions;

    /** Simulator for game state predictions */
    private Simulator gameSimulator;

    /** Sequence of actions selected during tree traversal */
    private LinkedList<Action> selectedActionSequence;

    /** Original HP of the player before simulation */
    private int initialPlayerHp;

    /** Original HP of the opponent before simulation */
    private int initialOpponentHp;

    /** Flag indicating if this node has been expanded */
    private boolean isExpanded;

    /** Action sequence for the player during simulation */
    private Deque<Action> playerActionSequence;
    
    /** Action sequence for the opponent during simulation */
    private Deque<Action> opponentActionSequence;

    /** Motion data for the player character */
    private ArrayList<MotionData> playerMotionData;
    
    /** Motion data for the opponent character */
    private ArrayList<MotionData> opponentMotionData;
    
    /** List of air actions */
    private List<Action> airActions;
    
    /** List of ground actions */
    private List<Action> groundActions;
    
    /** Special skill action */
    private Action specialMoveAction;
    
    /** Attributes for game control */
    private FrameData frameData;
    private boolean isPlayerOne;    
    private CommandCenter commandCenter;
    private GameData gameData;

    /**
     * Constructs a new MCTSNode with the specified parameters.
     * 
     * @param frameData the current frame data
     * @param parentNode the parent node in the search tree
     * @param playerActions available actions for the player
     * @param opponentActions available actions for the opponent
     * @param gameData the game data
     * @param isPlayerOne whether this node represents player one
     * @param commandCenter the command center for action execution
     * @param actionSequence the sequence of actions selected so far
     */
    public MCTSNode(FrameData frameData, MCTSNode parentNode, LinkedList<Action> playerActions,
            LinkedList<Action> opponentActions, GameData gameData, boolean isPlayerOne,
            CommandCenter commandCenter, LinkedList<Action> actionSequence) {
        this(frameData, parentNode, playerActions, opponentActions, gameData, isPlayerOne, commandCenter);
        this.selectedActionSequence = actionSequence;
    }

    /**
     * Constructs a new MCTSNode with the specified parameters.
     * 
     * @param frameData the current frame data
     * @param parentNode the parent node in the search tree
     * @param playerActions available actions for the player
     * @param opponentActions available actions for the opponent
     * @param gameData the game data
     * @param isPlayerOne whether this node represents player one
     * @param commandCenter the command center for action execution
     */
    public MCTSNode(FrameData frameData, MCTSNode parentNode, LinkedList<Action> playerActions,
            LinkedList<Action> opponentActions, GameData gameData, boolean isPlayerOne,
            CommandCenter commandCenter) {
        this.frameData = frameData;
        this.parentNode = parentNode;
        this.availablePlayerActions = playerActions;
        this.availableOpponentActions = opponentActions;
        this.gameData = gameData;
        this.gameSimulator = new Simulator(gameData);
        this.isPlayerOne = isPlayerOne;
        this.commandCenter = commandCenter;

        this.selectedActionSequence = new LinkedList<Action>();
        this.randomGenerator = new Random();
        this.playerActionSequence = new LinkedList<Action>();
        this.opponentActionSequence = new LinkedList<Action>();

        CharacterData playerCharacter = frameData.getCharacter(isPlayerOne);
        CharacterData opponentCharacter = frameData.getCharacter(!isPlayerOne);
        initialPlayerHp = playerCharacter.getHp();
        initialOpponentHp = opponentCharacter.getHp();

        if (this.parentNode != null) {
            this.nodeDepth = this.parentNode.nodeDepth + 1;
        } else {
            this.nodeDepth = 0;
        }
        
        // Initialize data for legal action detection
        this.playerMotionData = gameData.getMotionData(isPlayerOne);
        this.opponentMotionData = gameData.getMotionData(!isPlayerOne);
        
        this.airActions = Util.getActions(Util.TYPE_OF_ACTION.AIR);
        this.groundActions = Util.getActions(Util.TYPE_OF_ACTION.GROUND);
        this.specialMoveAction = Action.STAND_D_DF_FC;
    }
    
    /**
     * Returns the legal actions for a player in the given state.
     *
     * @param frameData the current frame data
     * @param forPlayer true for the player, false for the opponent
     * @return List of legal actions
     */
    public LinkedList<Action> getLegalActions(FrameData frameData, boolean forPlayer) {
        return new LinkedList<>(
        		Util.getLegalActions(
        		        frameData.getCharacter(forPlayer),
        		        forPlayer ? playerMotionData : opponentMotionData,
        		        specialMoveAction,
        		        airActions,
        		        groundActions)
        		);
    }
    
    /**
     * Sets the available actions for the player.
     *
     * @param playerActions the list of available player actions
     */
    public void setMyActions(LinkedList<Action> playerActions) {
        this.availablePlayerActions = playerActions;
    }

    /**
     * Sets the available actions for the opponent.
     *
     * @param opponentActions the list of available opponent actions
     */
    public void setOppActions(LinkedList<Action> opponentActions) {
        this.availableOpponentActions = opponentActions;
    }

    /**
     * Executes the Monte Carlo Tree Search algorithm.
     *
     * @return the best action found by the search
     */
    public Action executeMCTS() {
        long startTime = System.nanoTime();
        for (int i = 0; System.nanoTime() - startTime <= UCT_MAX_TIME && i < UCT_ITERATION_LIMIT; i++) {
            executeUCT();
        }

        return getMostVisitedAction();
    } 

    /**
     * Performs a simulation (playout) from this node.
     *
     * @return the evaluation score from the simulation
     */
    public double simulate() {
        playerActionSequence.clear();
        opponentActionSequence.clear();

        for (int i = 0; i < selectedActionSequence.size(); i++) {
            playerActionSequence.add(selectedActionSequence.get(i));
        }

        for (int i = 0; i < 5 - selectedActionSequence.size(); i++) {
            playerActionSequence.add(availablePlayerActions.get(randomGenerator.nextInt(availablePlayerActions.size())));
        }

        for (int i = 0; i < 5; i++) {
            opponentActionSequence.add(availableOpponentActions.get(randomGenerator.nextInt(availableOpponentActions.size())));
        }

        FrameData resultFrame = gameSimulator.simulate(frameData, isPlayerOne, playerActionSequence, opponentActionSequence, SIMULATION_DURATION);
        return evaluateState(resultFrame);
    }

    /**
     * Executes the UCT (Upper Confidence Bound for Trees) algorithm.
     *
     * @return the evaluation score from UCT
     */
    private double executeUCT() {
        MCTSNode selectedChild = null;
        double bestUCB = -99999;

        for (MCTSNode child : this.childNodes) {
            if (child.visitCount == 0) {
                child.ucbValue = 9999 + randomGenerator.nextInt(50);
            } else {
                child.ucbValue = calculateUCB1(child.accumulatedScore / child.visitCount, visitCount, child.visitCount);
            }

            if (bestUCB < child.ucbValue) {
                selectedChild = child;
                bestUCB = child.ucbValue;
            }
        }

        double score = 0;
        if (selectedChild.visitCount == 0) {
            score = selectedChild.simulate();
        } else {
            if (selectedChild.childNodes == null) {
                if (selectedChild.nodeDepth < UCT_MAX_DEPTH) {
                    if (UCT_EXPANSION_THRESHOLD <= selectedChild.visitCount) {
                        selectedChild.expandNode();
                        selectedChild.isExpanded = true;
                        score = selectedChild.executeUCT();
                    } else {
                        score = selectedChild.simulate();
                    }
                } else {
                    score = selectedChild.simulate();
                }
            } else {
                if (selectedChild.nodeDepth < UCT_MAX_DEPTH) {
                    score = selectedChild.executeUCT();
                } else {
                    score = selectedChild.simulate();
                }
            }
        }

        selectedChild.visitCount++;
        selectedChild.accumulatedScore += score;

        if (nodeDepth == 0) {
            visitCount++;
        }

        return score;
    }

    /**
     * Expands this node by creating child nodes for all available actions.
     */
    public void expandNode() {
        this.childNodes = new MCTSNode[availablePlayerActions.size()];

        for (int i = 0; i < childNodes.length; i++) {
            LinkedList<Action> newActionSequence = new LinkedList<Action>();
            for (Action act : selectedActionSequence) {
                newActionSequence.add(act);
            }

            newActionSequence.add(availablePlayerActions.get(i));

            childNodes[i] = new MCTSNode(frameData, this, availablePlayerActions, availableOpponentActions, 
                                       gameData, isPlayerOne, commandCenter, newActionSequence);
        }
    }

    /**
     * Returns the most visited action from the child nodes.
     *
     * @return the most visited action
     */
    private Action getMostVisitedAction() {
        int selectedIndex = -1;
        double bestVisitCount = -9999;

        for (int i = 0; i < childNodes.length; i++) {
            if (MCTSAgent.DEBUG_ENABLED) {
                System.out.println("Score:" + childNodes[i].accumulatedScore / childNodes[i].visitCount + ",Visits:"
                    + childNodes[i].visitCount + ",UCB1:" + childNodes[i].ucbValue + ",Action:" + availablePlayerActions.get(i));
            }

            if (bestVisitCount < childNodes[i].visitCount) {
                bestVisitCount = childNodes[i].visitCount;
                selectedIndex = i;
            }
        }

        if (MCTSAgent.DEBUG_ENABLED) {
            System.out.println(availablePlayerActions.get(selectedIndex) + ",Total visits:" + visitCount);
            System.out.println("");
        }

        return this.availablePlayerActions.get(selectedIndex);
    }

    /**
     * Evaluates the game state based on HP difference.
     *
     * @param frameData the frame data to evaluate
     * @return the evaluation score
     */
    private int evaluateState(FrameData frameData) {
        return (frameData.getCharacter(isPlayerOne).getHp() - initialPlayerHp) - 
               (frameData.getCharacter(!isPlayerOne).getHp() - initialOpponentHp);
    }

    /**
     * Calculates the UCB1 value for a node.
     *
     * @param averageScore the average score of the node
     * @param totalVisits the total number of visits to the parent node
     * @param nodeVisits the number of visits to this node
     * @return the UCB1 value
     */
    private double calculateUCB1(double averageScore, int totalVisits, int nodeVisits) {
        return averageScore + UCB1_EXPLORATION_CONSTANT * Math.sqrt((2 * Math.log(totalVisits)) / nodeVisits);
    }

    /**
     * Prints the node structure for debugging purposes.
     *
     * @param node the node to print
     */
    public void printNodeStructure(MCTSNode node) {
        System.out.println("Total visits:" + node.visitCount);
        for (int i = 0; i < node.childNodes.length; i++) {
            System.out.println(i + ",Visits:" + node.childNodes[i].visitCount + ",Depth:" + node.childNodes[i].nodeDepth
                + ",score:" + node.childNodes[i].accumulatedScore / node.childNodes[i].visitCount + ",ucb:"
                + node.childNodes[i].ucbValue);
        }
        System.out.println("");
        for (int i = 0; i < node.childNodes.length; i++) {
            if (node.childNodes[i].isExpanded) {
                printNodeStructure(node.childNodes[i]);
            }
        }
    }
    
    /**
     * Returns the Q-values for all possible actions.
     *
     * @return array of Q-values for all actions
     */
    public double[] getQValues() {
        double[] actionValues = new double[Action.values().length];
        Arrays.fill(actionValues, 0.0);
        
        if (childNodes != null && childNodes.length > 0) {
            for (int i = 0; i < childNodes.length; i++) {
                if (childNodes[i] != null && childNodes[i].visitCount > 0) {
                    Action action = availablePlayerActions.get(i);
                    actionValues[action.ordinal()] = childNodes[i].accumulatedScore / childNodes[i].visitCount;
                }
            }
        }
        
        return actionValues;
    }
}
