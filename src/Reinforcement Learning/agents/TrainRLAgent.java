import java.io.IOException;
import struct.GameData;

/**
 * TrainRLAgent extends BaseRLAgent and is employed for the first 
 * training methodology (MCTS:Self-Play), which is  divided in two phases:
 * 		1. Training against MCTS agent
 * 		2. Self-Play training starting from the weights obtained in phase 1.
 */
public class TrainRLAgent extends BaseRLAgent {
	
	private static final String PRETRAINED_WEIGHTS_FOLDER_PH1 = "pesi/selfPlay/fase1/";
	private static final String WEIGHTS_FOLDER_PH2 = "pesi/selfPlay/fase2/";
	
	private static boolean IS_PRETRAINED = true;

	// Weights from the first training phase (MCTS)
	private static final String PRETRAINED_WEIGHTS = PRETRAINED_WEIGHTS_FOLDER_PH1 + "weights_445.dat";
	
	private double bestAverageReward = Double.NEGATIVE_INFINITY;
	
	/**
	 * Initializes the agent by calling the superclass initialization
 	 * and, if the first training phase has been completed, loads the 
 	 * weights obtained during that phase.
     *
     * @param gd GameData object containing the game’s global information
     * @param playerNumber ID of the controlled player (true = P1, false = P2)
     * @return 0
     */
	@Override
	public int initialize(GameData gd, boolean playerNumber) {
		super.initialize(gd, playerNumber);

		if(IS_PRETRAINED) {
			try {
				this.getLinearQLearning().loadWeightsFromFile(PRETRAINED_WEIGHTS);
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		return 0;
	}
	
	/**
     * Called at the end of each round.
	 * 
	 * This method works as follows:
     *
     *   - Computes a bonus reward based on whether the agent won or lost the round.
     *   - Logs episode results (health points, reward, timesteps).
     *   - Every 5 episodes, computes the average reward over the last 5 rounds
     *       and saves the agent’s weights if this average improves by at least 5%
     *       over the best recorded so far.
     *   - Appends the total reward to the training history and updates the learning
     *       rate (after 300 episodes).
     *   - Resets the agent’s internal state for the next round.
     *
     * @param p1Hp remaining HP of player 1
     * @param p2Hp remaining HP of player 2
     * @param frames the number of frames played in the round
     */
	@Override
	public void roundEnd(int p1Hp, int p2Hp, int frames) {
		// Bonus reward to encourage winning
		double reward = (this.isPlayerNumber() && p1Hp > p2Hp) || (!this.isPlayerNumber() && p2Hp > p1Hp) ? 10.0 : -10.0;; // in maniera da incoraggiare la vittoria
		this.setTotalReward(this.getTotalReward() + reward);
		
		
		System.out.println(p1Hp + " " + p2Hp + " " + frames);
		System.out.println("Episodio: " + this.getRoundNum() + " Total Reward: " + this.getTotalReward() + " TimeSteps: " + this.getTimeSteps());
		
		String dir = "log";

		// If the first training phase has been completed, save the output from self-play; 
		// otherwise, save the results from the initial training phase.
		String fileName = (IS_PRETRAINED) ? "output_self_play.csv" : "output.csv";
		
		Util.logRLData(dir, fileName, p1Hp, p2Hp, frames, this.getTimeSteps(), this.getRoundNum(), this.getTotalReward());
		
		int windowSize = 5;
		double improvementThreshold = 0.05;
		
		// Every 5 episodes, evaluate whether to save weights
		if (this.getRoundNum() % windowSize == 0) {
		    // Computes the average of the last windowSize rewards
		    double currentAverageReward = this.getLinearQLearning().getRewardData()
		        .subList(Math.max(0, this.getLinearQLearning().getRewardData().size() - windowSize), 
		                this.getLinearQLearning().getRewardData().size())
		        .stream()
		        .mapToDouble(Double::doubleValue)
		        .average()
		        .orElse(Double.NEGATIVE_INFINITY);
		    
		    // Save weights only if performance improved significantly
		    if (currentAverageReward > bestAverageReward * (1 + improvementThreshold)) {
		        bestAverageReward = currentAverageReward;
		        try {
		        	String weightsFile = (IS_PRETRAINED) ? (WEIGHTS_FOLDER_PH2 + "weights_self_play.dat") : (PRETRAINED_WEIGHTS_FOLDER_PH1 + "weights_" + this.getRoundNum() + ".dat");
					this.getLinearQLearning().saveWeightsToFile(weightsFile);
				
		        } catch (IOException e) {
					e.printStackTrace();
				}
		        System.out.println("Pesi salvati all'episodio " + this.getRoundNum() + 
		                         " con reward media: " + currentAverageReward);
		    }
		    
		}
		
		this.getLinearQLearning().addReward(this.getTotalReward());
		this.getLinearQLearning().updateLearningRate(this.getRoundNum()); // Adjust learning rate after 300 episodes
		
		this.reset();
		
	}
}
