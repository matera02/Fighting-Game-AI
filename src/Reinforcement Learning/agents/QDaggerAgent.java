import java.util.Arrays;
import java.util.LinkedList;

import enumerate.Action;
import simulator.Simulator;
import struct.FrameData;
import struct.GameData;

/**
 * QDaggerAgent extends BaseRLAgent and implements a RL agent
 * that uses the QDagger algorithm, combining TD learning with
 * policy distillation using MCTS as teacher.
 */
public class QDaggerAgent extends BaseRLAgent {

    private static final int FRAME_AHEAD = 14; //* Number of frames to simulate ahead for MCTS planning */

    private GameData gameData;
    private Simulator simulator;

    private QDagger<FrameData> qDagger;

    private int numWins = 0;
    private Util.Timer timer;

	/**
     * Initializes the agent with all required components.
	 * Sets up the MCTS teacher policy and initializes the QDagger algorithm.
	 * 
     * @param gd GameData object containing the game’s global information
     * @param playerNumber ID of the controlled player (true = P1, false = P2)
     * @return 0
     */
    @Override
    public int initialize(GameData gd, boolean playerNumber) {
        this.gameData = gd;
        super.initialize(this.gameData, playerNumber);
        this.simulator = this.gameData.getSimulator();
        this.timer = new Util.Timer();

        // Expert policy using MCTS
        QDagger.TeacherPolicy<FrameData> teacherPolicy = new QDagger.TeacherPolicy<FrameData>() {

			/**
             * Gets Q-values from MCTS expert for the given state.
             * Performs MCTS planning to estimate expert Q-values.
             *
             * @param state the current frame data (= state)
             * @return array of Q-values for all actions estimated by MCTS
             */
            @Override
            public double[] getQValues(FrameData state) {
                if (state.getEmptyFlag()) {
                    double[] empty = new double[Action.values().length];
                    Arrays.fill(empty, 0.0);
                    return empty;
                }

				// Simulate ahead 
                FrameData simulatorAheadFrameData = simulator.simulate(state, playerNumber, null, null, FRAME_AHEAD);
                
				// Create root node for MCTS
                MCTSNode rootNode = new MCTSNode(
                    simulatorAheadFrameData,
                    null,
                    new LinkedList<Action>(),
                    new LinkedList<Action>(),
                    gameData,
                    playerNumber,
                    getCc()
                );
				
                // Get legal actions using Node's built-in method
                LinkedList<Action> myLegalActions = rootNode.getLegalActions(simulatorAheadFrameData, playerNumber);
                LinkedList<Action> oppLegalActions = rootNode.getLegalActions(simulatorAheadFrameData, !playerNumber);
                
                // Set actions in the node
                rootNode.setMyActions(myLegalActions);
                rootNode.setOppActions(oppLegalActions);
                
                // Create child nodes and execute MCTS
                rootNode.expandNode();
                rootNode.executeMCTS();
                
                return rootNode.getQValues();
            }
        };

        // Initialize QDagger
        qDagger = new QDagger<>(this.getLinearQLearning(), teacherPolicy, 1.0, 0.99, Action.values().length);
        
        return 0;
    }

	/**
	 * Executes a single RL step.
	 * In contrast to the implementation in the superclass, 
	 * this method invokes the update method of QDagger algorithm.
	 */
    @Override
    protected void runRL() {
        this.timer.updateTimer();

        double[] currentState = this.getObservation();
        int action = this.getLinearQLearning().selectAction(currentState);

        Environment.StepResult result = this.getEnv().step(this.getFrameData(), Action.values()[action]);

        double reward = result.getReward();
        double[] newState = result.getObservation();

        // QDagger update
        qDagger.update(this.getFrameData(), currentState, action, reward, newState);

        this.setObservation(newState);
        this.setTotalReward(this.getTotalReward() + reward);
        this.setTimeSteps(this.getTimeSteps() + 1);
    }

    @Override
    public void close() {
        System.out.println("game end");
        System.out.println("Vittorie: " + this.numWins + " Rounds: " + this.getRoundNum());
    }

	/**
     * Called at the end of each round.
	 * Handles reward calculation, updates win statistics, 
	 * and logs results.
     *
     * @param p1Hp remaining HP of player 1
     * @param p2Hp remaining HP of player 2
     * @param frames the number of frames played in the round
     */
    @Override
    public void roundEnd(int p1Hp, int p2Hp, int frames) {
        double reward = (this.isPlayerNumber() && p1Hp > p2Hp) || (!this.isPlayerNumber() && p2Hp > p1Hp) ? 10.0 : -10.0;
        if (reward == 10.0) ++this.numWins;
        this.setTotalReward(this.getTotalReward() + reward);

        System.out.println(p1Hp + " " + p2Hp + " " + frames);
        System.out.println("Episodio: " + this.getRoundNum() +
                " Total Reward: " + this.getTotalReward() +
                " TimeSteps: " + this.getTimeSteps());

        String dir = "log";
        String fileName = "test_qdagger.csv";

        Util.logRLData(dir, fileName,
                p1Hp, p2Hp, frames,
                this.getTimeSteps(), this.getRoundNum(),
                this.getTotalReward(), this.timer.getTimerSeconds());

        this.getLinearQLearning().addReward(this.getTotalReward());

        super.reset();
        this.timer.resetTimer();
    }
}
