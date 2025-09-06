import aiinterface.AIInterface;
import struct.FrameData;
import struct.GameData;
import struct.Key;

/**
 * SelfPlayAgent is used as the opponent during the second phase of training for TrainRLAgent.
 * 
 * This class alternates between two agents an MCTS agent and a RL agent in a 1:3 ratio 
 * (1 round with MCTS, 3 rounds with RL). 
 * Every 500 episodes, the number of rounds for both agents is progressively increased.
 * 
 * The RLAgent used here periodically updates its weights from a shared file. This file
 * is maintained by TrainRLAgent during training, ensuring that the SelfPlayAgent
 * always uses the latest representation of the trained RL agent.
 */
public class SelfPlayAgent implements AIInterface {
	 // Shared weights file for the second training phase
	private static final String WEIGHTS_FOLDER_PH2 = "pesi/selfPlay/fase2/";
	private static final String WEIGHTS = WEIGHTS_FOLDER_PH2 + "weights_self_play.dat"; // Initially copied from the first training phase, then updated periodically
	private static final int INCREMENT_ROUNDS = 500;  // Increase agents' rounds every 500 episodes	
	
	// Agents for Self-Play
	private MCTSAgent mctsAgent;
	private TestRLAgent rlAgent;
	private boolean isRunningMcts = true;
	
	// Round management
	private int roundNum = 1;
	private int maxRoundsMCTS = 1;
	private int maxRoundRL = 3;
	private int currentAvailableRoundsMCTS = 1;
	private int currentAvailableRoundsRL = 3;
	
	/**
	 * Initialize both MCTS and RL agents.
     *
     * @param gd GameData object containing the game’s global information
     * @param playerNumber ID of the controlled player (true = P1, false = P2)
     * @return 0
     */
	@Override
	public int initialize(GameData gd, boolean playerNumber) {
		
		this.mctsAgent = new MCTSAgent();
		this.mctsAgent.initialize(gd, playerNumber);
		
		this.rlAgent = new TestRLAgent();
		this.rlAgent.initialize(gd, playerNumber);
		
		return 0;
	}

	/**
	 * Get the information about the game state at each frame.
	 * This method is managed for both agents. 
	 * 
	 *  @param frameData object of the FrameData class 
	 * 			containing information about the match
	 */
	@Override
	public void getInformation(FrameData fd, boolean arg1) {
		if(isRunningMcts) {
			this.mctsAgent.getInformation(fd, arg1);
		} else {
			this.rlAgent.getInformation(fd);
		}
	}
	
	/**
	 * This method is called every frame.
	 * Alternate processing method between agents 
	 * based on round availability. 
	 */
	@Override
	public void processing() {
		if(isRunningMcts) {
			this.mctsAgent.processing();
		} else {
			this.rlAgent.processing();
		}
	}

	/**
     * Returns the current key input for the FightingICE engine
	 * based on running agent.
     *
     * @return Key object representing the agent's input
     */
	@Override
	public Key input() {
		return isRunningMcts ? this.mctsAgent.input() : this.rlAgent.input();
	}
	
	@Override
	public void close() {
		System.out.println("game end");
	}

	/**
     * Called at the end of each round.
	 * 
     * @param p1Hp remaining HP of player 1
     * @param p2Hp remaining HP of player 2
     * @param frames the number of frames played in the round
     */
	@Override
	public void roundEnd(int p1Hp, int p2Hp, int frames) {
		
		// Decrement the number of available rounds for the agent that just ran
		if(isRunningMcts) {
            //System.out.println("Round " + roundNum + ": MCTS (rimangono " + 
            //                 (currentAvailableRoundsMCTS-1) + " round MCTS)");
            
            --currentAvailableRoundsMCTS;
        } else {
            //System.out.println("Round " + roundNum + ": RL (rimangono " + 
            //                 (currentAvailableRoundsRL-1) + " round RL)");
            
            --currentAvailableRoundsRL;
        }

        //System.out.println("Risultati round - P1 HP: " + p1Hp + 
        //                 ", P2 HP: " + p2Hp + ", Frames: " + frames);
		
		// Increase the number of rounds assigned to each agent every INCREMENT_ROUNDS episodes
        this.incrementAgents(INCREMENT_ROUNDS);
		
		// Decide which agent should run next
		this.chooseAgent();

		// Advance the global round counter
		this.roundNum++;
		
		// Update RL agent weights from the shared file produced by TrainRLAgent
		// (TrainRLAgent periodically saves the best weights to this file)
		this.upgradeRLAgentWeights(WEIGHTS);
		
	}

	/**
     * Increments the maximum number of rounds available for both MCTS and RL agents
     * after a fixed interval of episodes.
     *
     * @param maxRounds number of episodes after which to increase agent rounds
     */
	private void incrementAgents(int maxRounds) {
		if((roundNum % maxRounds) == 0) {
			this.maxRoundsMCTS += 1;
			this.maxRoundRL += 3;
			resetCurrentRounds();
		}
	}

	/**
     * Resets the available rounds for both agents to their maximum values.
     * Always restarts with MCTS.
     */
	private void resetCurrentRounds() {
        currentAvailableRoundsMCTS = maxRoundsMCTS;
        currentAvailableRoundsRL = maxRoundRL;
        isRunningMcts = true;
    }
	
	/**
     * Chooses the agent to run based on the number of available 
	 * rounds left for each agent.
     */
	private void chooseAgent() {
		
		if(this.currentAvailableRoundsMCTS == 0 && this.currentAvailableRoundsRL > 0) {
			isRunningMcts = false;
			return;
		}
		
		if(this.currentAvailableRoundsMCTS == 0 && this.currentAvailableRoundsRL == 0) {
			resetCurrentRounds();
		}
		
		
		//System.out.println("Stato corrente - MCTS rounds: " + currentAvailableRoundsMCTS + 
        //        ", RL rounds: " + currentAvailableRoundsRL);
		
	}
	
	/**
     * Updates the RL agent's weights from the shared file.
     *
     * @param fileName path to the weights file
     */
	private void upgradeRLAgentWeights(String fileName) {
		this.rlAgent.setWeights(fileName);
	}

	@Override
	public void getInformation(FrameData fd) {
		// Not used in this implementation
		
	}
}
