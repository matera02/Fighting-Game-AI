import aiinterface.AIInterface;
import aiinterface.CommandCenter;
import struct.FrameData;
import struct.GameData;
import struct.Key;
import enumerate.Action;

/**
 * This class is used to test the Environment class.
 */
public class TestEnvironment implements AIInterface {
	
	private boolean playerNumber;
	private FrameData frameData;
	private Key key;
	private CommandCenter cc;
	private Environment env;
	private int roundNum;
	private boolean isReset = false;
	
	@Override
	public int initialize(GameData gd, boolean playerNumber) {
		this.playerNumber = playerNumber;
		this.frameData = new FrameData();
		this.key = new Key();
		this.cc = new CommandCenter();
		this.roundNum = 0;
		
		this.env = new Environment(frameData, cc, playerNumber);
		
		return 0;
	}

	@Override
	public void getInformation(FrameData fd) {
		this.frameData = fd;
		this.cc.setFrameData(frameData, playerNumber);
	}

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
			
			double[] observation;
			
			if(!isReset) {
				observation = env.reset();
				Action action = Util.getRandomAction(Util.TYPE_OF_ACTION.GROUND);
				cc.commandCall(action.name());
				System.out.println("SONO IN RESET");
				this.isReset = true;
			} else {
				System.out.println("SONO IN STEP");
				Action action = Util.getRandomAction(Util.TYPE_OF_ACTION.GROUND);
				Environment.StepResult res = env.step(frameData, action);
				observation = res.getObservation();
				double reward = res.getReward();
				System.out.println("Reward: " + reward);
			}
			
			//System.out.println("Dimensione osservazioni: " + observation.length);
			//voglio stampare le info sulle osservazioni
		}
	}

	@Override
	public Key input() {
		return this.key;
	}

	@Override
	public void close() {
		System.out.println("game end");
	}

	@Override
	public void roundEnd(int p1Hp, int p2Hp, int frames) {
		System.out.println(p1Hp + " " + p2Hp + " " + frames);
		System.out.println(this.roundNum);
		
		/*for(int i = 0; i < this.lastObservation.length; ++i) {
			System.out.println(this.lastObservation[i]);
		}*/
		
		env.close();
		
		this.isReset = false;
		this.frameData = new FrameData();
		this.roundNum++;
	}

}
