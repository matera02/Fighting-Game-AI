import java.io.IOException;
import enumerate.Action;
import struct.GameData;

/**
 * TestRLAgent is a sublcass of BaseRLAgent used for testing
 * different training strategies for RL agents.
 */
public class TestRLAgent extends BaseRLAgent {

	// Path to the weights file used for testing
	//private final String WEIGHTS_FILE = "pesi/selfPlay/fase2/weights_self_play.dat"; // weights after training with self-play
	private final String WEIGHTS_FILE = "pesi/geneticRL/best_weights_gen9_ep450.dat"; // weights after training with evoRL

	private Util.Timer timer; // Timer used for logging statistics
	private int numWins = 0; // Count of wins across rounds
	
	/**
	 * Initializes the agent by calling the superclass initialization and 
	 * loads the weights obtained from the training phase from file.
	 * 
	 * @param gd GameData object containing the game’s global information
     * @param playerNumber ID of the controlled player (true = P1, false = P2)
     * @return 0 
	 */
	@Override
	public int initialize(GameData gd, boolean playerNumber) {
		super.initialize(gd, playerNumber);
		
		try {
			this.getLinearQLearning().loadWeightsFromFile(WEIGHTS_FILE);
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		this.timer = new Util.Timer(); // timer for measuring game statistics.
		
		return 0;
	}
	
	/**
	 * Executes a single RL step.
	 * 
	 * In contrast to BaseRLAgent, this version does not update
	 * the underlying Linear Q-learning model.
	 */
	@Override
	protected void runRL() {
		this.timer.updateTimer();
		
		int action = this.getLinearQLearning().selectAction(this.getObservation());
		Environment.StepResult result = this.getEnv().step(this.getFrameData(), Action.values()[action]);
		double reward = result.getReward();
		double[] newObservation = result.getObservation();
		
		this.setObservation(newObservation);
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
	 * 
	 * 		- Computes a bonus reward based on whether the agent won or lost the round.
	 * 		- Updates the total reward and the win counter.
	 * 		- Logs statistics to a CSV file.
	 * 		- Call superclass reset method and reset timer.
	 * 
     * @param p1Hp remaining HP of player 1
     * @param p2Hp remaining HP of player 2
     * @param frames the number of frames played in the round
	 */
	@Override
	public void roundEnd(int p1Hp, int p2Hp, int frames) {
		
		double reward = (this.isPlayerNumber() && p1Hp > p2Hp) || (!this.isPlayerNumber() && p2Hp > p1Hp) ? 10.0 : -10.0;; // in maniera da incoraggiare la vittoria
		if(reward == 10.0) ++this.numWins;	
		this.setTotalReward(this.getTotalReward() + reward);
		
		
		System.out.println(p1Hp + " " + p2Hp + " " + frames);
		System.out.println("Episodio: " + this.getRoundNum() + " Total Reward: " + this.getTotalReward() + " TimeSteps: " + this.getTimeSteps());
		
		String dir = "log";
		String fileName = "test_garl.csv";
		
		Util.logRLData(dir, fileName, p1Hp, p2Hp, frames, this.getTimeSteps(), this.getRoundNum(), this.getTotalReward(), this.timer.getTimerSeconds());
		
		this.getLinearQLearning().addReward(this.getTotalReward());
		
		super.reset();
		this.timer.resetTimer();
	}
	
	
	/**
	 * Loads new weights into the agent.
	 * 
	 * This method is called in SelfPlayAgent to refresh
	 * the RL agent's weights during self-play.
	 *
	 * @param fileName path to the weights file
	 */
	public void setWeights(String fileName) {
		try {
			this.getLinearQLearning().loadWeightsFromFile(fileName);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
