import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import aiinterface.AIInterface;
import aiinterface.CommandCenter;
import enumerate.Action;
import simulator.Simulator;
import struct.CharacterData;
import struct.FrameData;
import struct.GameData;
import struct.Key;
import struct.MotionData;

/**
 * Agent implementation using Monte Carlo Tree Search (MCTS) 
 * algorithm for decision making .
 */
public class MCTSAgent implements AIInterface {

	private Simulator sim;
	private Key k;
	private CommandCenter cc;
	private boolean player;
	private GameData gd;
	private FrameData fd;
	private FrameData simAhead;
	private LinkedList<Action> myActions;
	private LinkedList<Action> opponentActions;
	private CharacterData myCharacterData;
	private CharacterData oppCharacterData;
	private static final int FRAME_AHEAD = 14;
	private ArrayList<MotionData> myMotionData;
	private ArrayList<MotionData> oppMotionData;
	private List<Action> airActions;
	private List<Action> groundActions;
	private Action spSkill;
	private MCTSNode rootNode;
	public static final boolean DEBUG_ENABLED = false;
	
	/**
     * Initializes the agent with all required components.
     *
     * @param gd GameData object containing the game’s global information
     * @param playerNumber ID of the controlled player (true = P1, false = P2)
     * @return 0
     */
	@Override
	public int initialize(GameData gameData, boolean playerNumber) {
		this.player = playerNumber;
		this.gd = gameData;

		this.k = new Key();
		this.fd = new FrameData();
		this.cc = new CommandCenter();

		this.myActions = new LinkedList<Action>();
		this.opponentActions = new LinkedList<Action>();

		sim = gameData.getSimulator();

		airActions = Util.getActions(Util.TYPE_OF_ACTION.AIR);
		groundActions = Util.getActions(Util.TYPE_OF_ACTION.GROUND);
		spSkill = Action.STAND_D_DF_FC;

		myMotionData = gameData.getMotionData(this.player);
		oppMotionData = gameData.getMotionData(!this.player);

		return 0;
	}
	
	/**
     * Returns the current key input for the FightingICE engine.
     *
     * @return Key object representing the agent's input
     */
	@Override
	public Key input() {
		return k;
	}
	
	/**
     * This method is called every frame.
     *
     * - If the game is not running or the agent is already performing an action, 
     *   the method exits early.
	 * 
     * - Otherwise the agent selects an action using MCTS.
     */
	@Override
	public void processing() {
		
		//if cannot process
		if (fd.getEmptyFlag() || fd.getFramesNumber() <= 0) {
			return;
		}
		
		if (cc.getSkillFlag()) {
			k = cc.getSkillKey();
		} else {
			k.empty();
			cc.skillCancel();
			
			this.prepare();
			
			rootNode = new MCTSNode(simAhead, null, myActions, opponentActions, gd, player, cc);	
			rootNode.expandNode();
			Action bestAction = rootNode.executeMCTS();
			if (DEBUG_ENABLED) {
				rootNode.printNodeStructure(rootNode);
			}
			cc.commandCall(bestAction.name());
		}
	}
	
	/**
     * Prepares for MCTS by simulating ahead and determining available actions.
     */
	private void prepare() {
		simAhead = sim.simulate(fd, player, null, null, FRAME_AHEAD);
		myCharacterData = simAhead.getCharacter(player);
		oppCharacterData = simAhead.getCharacter(!player);
		
		setMyAction();
		setOpponentAction();
	}

	
	private void setMyAction() {
		myActions.clear();
		myActions.addAll(Util.getLegalActions(myCharacterData, myMotionData, spSkill, airActions, groundActions));
	}

	
	private void setOpponentAction() {
		opponentActions.clear();
		opponentActions.addAll(Util.getLegalActions(oppCharacterData, oppMotionData, spSkill, airActions, groundActions));
	}
	
	/**
	 * Get the information about the game state at each frame. 
	 * 
	 *  @param frameData object of the FrameData class 
	 * 			containing information about the match
	 */
	@Override
	public void getInformation(FrameData fd) {
		this.fd = fd;
		this.cc.setFrameData(fd, player);
		
		myCharacterData = this.fd.getCharacter(player);
		oppCharacterData = this.fd.getCharacter(!player);
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
		// do nothing
	}
	
	/**
     * Cleans up resources at the end of the match. 
     */
	@Override
	public void close() {
		// do nothing
	}
}
