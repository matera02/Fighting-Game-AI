import aiinterface.AIInterface;
import aiinterface.CommandCenter;
import enumerate.Action;
import struct.FrameData;
import struct.GameData;
import struct.Key;

/**
 * Abstract class for RL agents implementing AIInterface.
 * 
 * This class provides a common structure for agents that use LinearQLearning
 * and Environment. 
 * 
 * It defines the initialization and core loop logic, while leaving 
 * the round-end behavior to subclasses.
 *
 * Utility methods, getters, and setters are also included.
 */
public abstract class BaseRLAgent implements AIInterface {
	
	private boolean playerNumber;
	private FrameData frameData;
	private Key key;
	private CommandCenter cc;
	
	private Environment env;
	private LinearQLearning linearQLearning;
	
	private int roundNum;
	private boolean isReset = false;
	private double totalReward = 0.0;
	private double[] observation;
	private int timeSteps = 0;	

	/**
     * Initializes the agent with all required components.
	 * 
     * This method sets up the environment, Linear Q-Learning algorithm, and other
     * essential members of the class. Subclasses should call super.initialize()
     * and may add additional setup logic such as loading weights from a file. 
     *
     * @param gd GameData object containing the game’s global information
     * @param playerNumber ID of the controlled player (true = P1, false = P2)
     * @return 0
     */
	@Override
	public int initialize(GameData gd, boolean playerNumber) {
		this.playerNumber = playerNumber;
		this.frameData = new FrameData();
		this.key = new Key();
		this.cc = new CommandCenter();
		this.roundNum = 1;
		
		this.env = new Environment(frameData, cc, playerNumber);
		this.linearQLearning = new LinearQLearning(Environment.OBSERVATION_SPACE_SIZE, Environment.ACTION_SPACE_SIZE);
		this.observation = new double[Environment.OBSERVATION_SPACE_SIZE];
		
		// In subclasses: call super and then load weights from file if needed
		
		return 0;
	}

	/**
	 * Get the information about the game state at each frame. 
	 * 
	 *  @param frameData object of the FrameData class 
	 * 			containing information about the match
	 */
	@Override
	public void getInformation(FrameData fd) {
		this.frameData = fd;
		this.cc.setFrameData(frameData, playerNumber);
	}

	/**
     * This method is called every frame.
     *
     * - If the game is not running or the agent is already performing an action, 
     *   the method exits early.
	 * 
     * - Otherwise, it works as follows:
     *
     *     - It ensures the environment has been reset.
	 * 	   - It executes one RL step.
	 * 
     */
	@Override
	public void processing() {
		if (frameData.getEmptyFlag() || frameData.getFramesNumber() <= 0) {
			return;
		}
		
		if (cc.getSkillFlag()) {
			key = cc.getSkillKey();
		} else {
			key.empty();
			cc.skillCancel();
			
			if(!isReset) {
				this.observation = env.reset();
				this.isReset = true;
			}
			
			runRL();
		}
	}

	/**
     * Returns the current key input for the FightingICE engine.
     *
     * @return Key object representing the agent's input
     */
	@Override
	public Key input() {
		return this.key;
	}
	
	@Override
	public void close() {
		System.out.println("game end");
	}

	/**
     * Called at the end of each round.
	 * 
	 * This method is meant to be overridden in subclasses, as the logic may vary depending
     * on the agent (e.g., saving weights, resetting counters, or performing evaluation).
     *
     * @param p1Hp remaining HP of player 1
     * @param p2Hp remaining HP of player 2
     * @param frames the number of frames played in the round
     */
	@Override
	public void roundEnd(int p1Hp, int p2Hp, int frames) {
		return;
	}
	
	/**
     * Executes a single RL step.
	 * 
     * The agent selects an action using Linear Q-Learning, applies it to the
     * environment, and updates the internal representation with the resulting 
	 * reward and new observation.
	 * 
     * After the update, the method stores the new observation, 
	 * accumulates the total reward, and increments the 
	 * time step counter.
     */
	protected void runRL() {
		int action = linearQLearning.selectAction(observation);
		Environment.StepResult result = env.step(frameData, Action.values()[action]);
		
		double reward = result.getReward();
		double[] newObservation = result.getObservation();
		
		// Do not call update during test; QDagger agent uses a different update logic
		linearQLearning.update(observation, action, reward, newObservation);
		
		this.observation = newObservation; 
		this.totalReward += reward;
		
		++this.timeSteps;
	}
	
	/**
     * Resets the agent’s internal state for a new round.
	 * 
     * This method resets the environment, frame data, reward counter, and time step counter.
     * It also increments the round number. Subclasses such as QDaggerAgent and TestRLAgent
     * may override this method to include additional logic (e.g., resetting timers).
     */
	protected void reset() {
		this.env.close();
		this.isReset = false;
		this.frameData = new FrameData();
		
		this.totalReward = 0.0;
		this.observation = new double[Environment.OBSERVATION_SPACE_SIZE];
		this.timeSteps = 0;
		this.roundNum++;
	}
	
	
	/** ------ Getters and setters ------ */

	protected boolean isPlayerNumber() {
		return playerNumber;
	}

	protected void setPlayerNumber(boolean playerNumber) {
		this.playerNumber = playerNumber;
	}

	protected FrameData getFrameData() {
		return frameData;
	}

	protected void setFrameData(FrameData frameData) {
		this.frameData = frameData;
	}

	protected Key getKey() {
		return key;
	}

	protected void setKey(Key key) {
		this.key = key;
	}

	protected CommandCenter getCc() {
		return cc;
	}

	protected void setCc(CommandCenter cc) {
		this.cc = cc;
	}

	protected Environment getEnv() {
		return env;
	}

	protected void setEnv(Environment env) {
		this.env = env;
	}

	protected LinearQLearning getLinearQLearning() {
		return linearQLearning;
	}

	protected void setLinearQLearning(LinearQLearning linearQLearning) {
		this.linearQLearning = linearQLearning;
	}

	protected int getRoundNum() {
		return roundNum;
	}

	protected void setRoundNum(int roundNum) {
		this.roundNum = roundNum;
	}

	protected boolean isReset() {
		return isReset;
	}

	protected void setReset(boolean isReset) {
		this.isReset = isReset;
	}

	protected double getTotalReward() {
		return totalReward;
	}

	protected void setTotalReward(double totalReward) {
		this.totalReward = totalReward;
	}

	protected double[] getObservation() {
		return observation;
	}

	protected void setObservation(double[] observation) {
		this.observation = observation;
	}

	protected int getTimeSteps() {
		return timeSteps;
	}

	protected void setTimeSteps(int timeSteps) {
		this.timeSteps = timeSteps;
	}
}
