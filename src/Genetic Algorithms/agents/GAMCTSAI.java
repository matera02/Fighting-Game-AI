import java.util.ArrayList;
import java.util.Deque;
import java.util.LinkedList;
import java.util.List;

import aiinterface.AIInterface;
import aiinterface.CommandCenter;
import enumerate.Action;
import enumerate.State;
import simulator.Simulator;
import struct.CharacterData;
import struct.FrameData;
import struct.GameData;
import struct.Key;
import struct.MotionData;

/**
 * GAMCTSAI implements a hybrid strategy, structured hierarchically by applying 
 * the Genetic Algorithm and the Monte Carlo Tree Search. 
 * 
 * The agent first evolves a sequence of promising actions using GA and then 
 * refines the decision by selecting the best action via MCTS simulations.
 */
public class GAMCTSAI implements AIInterface {
	
	public static final boolean DEBUG_ENABLED = false;

	private Simulator simulator;
	protected Key key;
	protected CommandCenter commandCenter;
	protected boolean playerNumber;
	protected GameData gameData;
	protected FrameData frameData;
	protected FrameData simulatorAheadFrameData;
	protected Deque<Action> myActions;
	protected Deque<Action> oppActions;
	private CharacterData myCharacter;
	private CharacterData oppCharacter;
	private static final int FRAME_AHEAD = 14;
	private ArrayList<MotionData> myMotion;
	private ArrayList<MotionData> oppMotion;
	private List<Action> actionAir;
	private List<Action> actionGround;
	private Action spSkill;
	protected MCTSNode rootNode;
	
	protected List<List<Double>> fitnessStats;
	protected Util.Timer timer;
	
	/**
     * Initializes the agent with all required components.
     *
     * @param gd GameData object containing the game’s global information
     * @param playerNumber ID of the controlled player (true = P1, false = P2)
     * @return 0
     */
	@Override
	public int initialize(GameData gd, boolean playerNumber) {
		this.playerNumber = playerNumber;
		this.gameData = gd;
		this.key = new Key();
		this.frameData = new FrameData();
		this.commandCenter = new CommandCenter();
		this.myActions = new LinkedList<Action>();
		this.oppActions = new LinkedList<Action>();
		this.simulator = gameData.getSimulator();
		
		this.actionAir = Util.getActions(Util.TYPE_OF_ACTION.AIR);
		this.actionGround = Util.getActions(Util.TYPE_OF_ACTION.GROUND);
		this.spSkill = Action.STAND_D_DF_FC;
		
		this.myMotion = gameData.getMotionData(this.playerNumber);
		this.oppMotion = gameData.getMotionData(!this.playerNumber);
		
		this.fitnessStats = new ArrayList<>();
		
		this.timer = new Util.Timer();
		
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
		this.commandCenter.setFrameData(frameData, playerNumber);
		
		myCharacter = this.frameData.getCharacter(playerNumber);
		oppCharacter = this.frameData.getCharacter(!playerNumber);
	}

	/**
     * This method is called every frame.
     *
     * - If the game is not running or the agent is already performing an action, 
     *   the method exits early.
	 * 
     * - Otherwise, it works as follows:
     *
     *     - Calls prepare() to initialize frame simulations and 
     *       determine the type of action for the GA's Chromosome.
	 * 
     *     - Runs the GeneticAlgorithm to generate a sequence of 
     *       promising actions, stored in myActions.
	 * 
     *     - Calls prepare() again to reset the environment for MCTS.
     *     - Initializes and executes MCTS to select the best 
     *       action based on simulations.
     *     - Executes the chosen action.
     */
	@Override
	public void processing() {
		this.timer.updateTimer();
		
		//if cannot process
		if (frameData.getEmptyFlag() || frameData.getFramesNumber() <= 0) {
			return;
		}
				
		if (commandCenter.getSkillFlag()) {
			key = commandCenter.getSkillKey();
		} else {
			key.empty();
			commandCenter.skillCancel();
			
			// preparation phase to execute the GA with the appropriate simulations
			Util.TYPE_OF_ACTION tp = this.prepare();
			//GeneticAlgorithm.DEBUG_MODE = true;
			GeneticAlgorithm ga = new GeneticAlgorithm(8, 20, 0.005, tp, this.frameData, this.gameData, this.playerNumber, this.oppActions); //related work params
			ga.run();
			// Use these actions for MCTS
			this.myActions = ga.getBestActions();

			// Take the statistics needed for evaluation
			this.fitnessStats.add(ga.getFitnessStats());

			this.prepare(); // I don’t need to know my action type, I have already determined the ones to start from
        	// I need to reinitialize everything for the simulations we will run with MCTS

			this.rootNode = new MCTSNode(this.simulatorAheadFrameData, null, 
					(LinkedList<Action>) this.myActions, (LinkedList<Action>) this.oppActions, 
					this.gameData, this.playerNumber, this.commandCenter);
			
			this.rootNode.expandNode();
			Action bestAction = this.rootNode.executeMCTS();
			
			if (DEBUG_ENABLED) {
				rootNode.printNodeStructure(rootNode);
			}
			
			commandCenter.commandCall(bestAction.name());
		}
	}

	/**
     * Returns the current key input for the FightingICE engine.
     *
     * @return Key object representing the agent's input
     */
	@Override
	public Key input() {
		return key;
	}

	/**
     * Cleans up resources at the end of the match. 
     */
	@Override
	public void close() {
		// do nothing
	}
	
	/**
     * Called at the end of each round: 
     * - Saves the fitness statistics and HP values to CSV files.
     * - Resets the timer and clears fitness data for the next round.
     *
     * @param p1Hp remaining HP of player 1
     * @param p2Hp remaining HP of player 2
     * @param frames the number of frames played in the round
     */
	@Override
	public void roundEnd(int p1Hp, int p2Hp, int frames) {
		System.out.println(p1Hp + " " + p2Hp + " " + frames);
		List<Integer> hp = List.of(p1Hp, p2Hp, this.timer.getTimerSeconds());
		
		String statsFolder = "stats";
		String fitnessFileName = "ftness.csv";
		String hpFileName = "hp.csv";
		
		Util.writeCsv(this.fitnessStats, statsFolder, fitnessFileName, true);
		Util.writeCsv(hp, statsFolder, hpFileName, true);
		
		this.fitnessStats.clear();
		this.timer.resetTimer();
	}
	
	/**
     * Prepares the agent for action selection.
	 * 
     * Simulates the game state 14 frames ahead, updates both characters' data, 
     * determines opponent actions via setOppAction(), and returns 
     * the type of action the agent is allowed to perform.
	 * 
     * This method is used both before running GA (to define chromosomes) 
     * and before running MCTS (to reset the simulated state). In the latter 
     * case, the returned action type is ignored.
     *
     * @return the type of action available (AIR, GROUND, or SKILL)
     */
	protected Util.TYPE_OF_ACTION prepare() {
		this.simulatorAheadFrameData = simulator.simulate(frameData, playerNumber, null, null, FRAME_AHEAD);
		this.myCharacter = this.simulatorAheadFrameData.getCharacter(playerNumber);
		this.oppCharacter = this.simulatorAheadFrameData.getCharacter(!playerNumber);
		this.setOppAction();
		return getMyTypeOfAction();
	}
	
	private Util.TYPE_OF_ACTION getMyTypeOfAction() {
		int energy = myCharacter.getEnergy();
		if(myCharacter.getState() == State.AIR) {
			return Util.TYPE_OF_ACTION.AIR;
		} else {	
			if (Math.abs(this.myMotion.get(Action.valueOf(spSkill.name()).ordinal())
					.getAttackStartAddEnergy()) <= energy) {
				return Util.TYPE_OF_ACTION.SKILL;
			}
			return Util.TYPE_OF_ACTION.GROUND;
		}
	}
	
	/**
     * Determines the set of valid actions the opponent can perform, based 
     * on its current state and available energy.
	 * 
     * Updates the oppActions deque with all feasible opponent moves.
     */
	private void setOppAction() {
		this.oppActions.clear();
		this.oppActions.addAll(Util.getLegalActions(oppCharacter, oppMotion, spSkill, actionAir, actionGround));
	}
}
