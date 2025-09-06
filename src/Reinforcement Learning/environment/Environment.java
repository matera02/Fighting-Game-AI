import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

import struct.FrameData;
import struct.CharacterData;
import java.util.Deque;
import struct.AttackData;
import java.util.Iterator;
import aiinterface.CommandCenter;
import enumerate.Action;

/**
 * The Environment class provides a RL environment wrapper 
 * for the fighting game framework. 
 * 
 * It defines the observation space, action space, and reward function used 
 * for the RL-based agents.
 * 
 * The environment supports the standard RL interface methods such as 
 * reset(), step(), and close().
 */
public class Environment {
	
	public static final int OBSERVATION_SPACE_SIZE = 126;
	public static final int ACTION_SPACE_SIZE = 56;
	
	private static final double LOW_OBSERVATION_SPACE_VALUE = -1.0;
	private static final double HIGH_OBSERVATION_SPACE_VALUE = 1.0;
	
	private double[] observation;
	private FrameData preFrameData = new FrameData();
	private FrameData frameData;
	private boolean playerNumber;
	private CommandCenter commandCenter;
	
	/**
	 * Constructor for the Environment class.
	 *
	 * @param frameData      current game frame data
	 * @param commandCenter  command center used to execute actions
	 * @param playerNumber   player identifier
	 */
	public Environment(FrameData frameData, CommandCenter commandCenter, boolean playerNumber) {
		this.frameData = frameData;
		this.observation = this.getObservation();
		this.commandCenter = commandCenter;
		this.playerNumber = playerNumber;
	}
	
	/**
	 * @return the current observation vector
	 */
	public double[] reset() {
		return this.observation;
	}
	
	/**
	 * Executes one step in the environment given the current game frame and an action.
	 * 
	 * This method updates the FrameData, executes the provided action through 
	 * the CommandCenter, computes the next observation, calculates the reward, 
	 * and returns a StepResult object containing the transition data.
	 *
	 * @param frameData updated game frame
	 * @param action    action to be executed
	 * @return a StepResult object containing the next observation, reward, done flag, and additional info
	 */
	public Environment.StepResult step(FrameData frameData, Action action) {
		this.setFrameData(frameData);

		this.commandCenter.commandCall(action.name());

		this.observation = this.getObservation();
		double reward = this.getReward();
		
		this.setPreFrameData(frameData);
		
		return new Environment.StepResult(this.observation, reward, false, null);
	}
	
	/**
	 * Container class that stores the result of a single step in the environment.
	 */
	public static class StepResult {
		
		private double[] observation;
		private double reward;
		private boolean done;
		private String info;
		
		public StepResult(double[] observation, double reward, boolean done, String info) {
			this.observation = observation;
			this.reward = reward;
			this.done = done;
			this.info = info;
		}
		
		
		public double[] getObservation() {
			return this.observation;
		}
		
		public double getReward() {
			return this.reward;
		}
		
		public boolean isDone() {
			return this.done;
		}
		
		public String getInfo() {
			return this.info;
		}
	}
	
	/**
	 * Closes the environment and clears stored frame data.  
	 * Additional cleanup may be required depending on resources used.
	 */
	public void close() {
		this.preFrameData = new FrameData();
		this.frameData = new FrameData();

	}
	
	
	public void setFrameData(FrameData frameData) {
		this.frameData = frameData;
	}
	
	public void setPreFrameData(FrameData newFrameData) {
		this.preFrameData = newFrameData;
	}
	
	/**
	 * @return the current observation vector from the game state.
	 * 
	 * Note: the uncommented observations are mapped between -1 and 1
	 */
	private double[] getObservation() {
		// If the frame is not initialized, return a zero observation vector
		if(this.preFrameData.getEmptyFlag() || this.frameData.getEmptyFlag()) return new double[OBSERVATION_SPACE_SIZE];
		
		List<Double> observation = new ArrayList<>();
		
		CharacterData my = this.frameData.getCharacter(this.playerNumber);
		CharacterData opp = this.frameData.getCharacter(!this.playerNumber);
		
		// RL-based agent information
		double myEnergy = my.getEnergy() / 300.0;
		double myX = ((my.getLeft() + my.getRight()) / 2.0 - 960.0/2.0) / (960.0/2.0);
		double myY = ((my.getBottom() + my.getTop()) / 2.0) / 640.0;
		double mySpeedX = my.getSpeedX() / 20.0;
		double mySpeedY = my.getSpeedY() / 28.0;
		
		int myState = my.getState().ordinal();
		int myAction = my.getAction().ordinal();
		
		double myRemainingFrame = my.getRemainingFrame() / 70.0;
		
		observation.add(myRemainingFrame);
		
		// Opponent information
		double oppEnergy = opp.getEnergy() / 300.0;
		double oppX = ((opp.getLeft() + opp.getRight()) / 2.0 - (my.getLeft() + my.getRight()) / 2.0) / 960.0;
		double oppY = ((opp.getBottom() + opp.getTop()) / 2.0) / 640.0;
		double oppSpeedX = opp.getSpeedX() / 20.0;  
		double oppSpeedY = opp.getSpeedY() / 28.0;
		
		int oppState = opp.getState().ordinal();
		int oppAction = opp.getAction().ordinal();
		
		double oppRemainingFrame = opp.getRemainingFrame() / 70.0;
		
		observation.add(oppRemainingFrame);
		
		// Add agent information
		observation.add(myEnergy);    // [0 , 1]
		observation.add(myX);        //  [-1, 1]
		observation.add(myY); 	    //   [0 , 1]
		observation.add(mySpeedX); //    [-1, 1]
		observation.add(mySpeedY);//     [-1, 1]
		
		// One-hot encoding for state and action
		this.addOneHotEncodedInfo(observation, 4, myState);
		this.addOneHotEncodedInfo(observation, 56, myAction);
		
		// Add opponent information
		observation.add(oppEnergy);    // [0 , 1]
		observation.add(oppX);        //  [-1, 1]
		observation.add(oppY); 	     //   [0 , 1]
		observation.add(oppSpeedX); //    [-1, 1]
		observation.add(oppSpeedY);//     [-1, 1]
		
		this.addOneHotEncodedInfo(observation, 4, oppState);
		this.addOneHotEncodedInfo(observation, 56, oppAction);
		
		// Projectile information
		Deque<AttackData> myProjectiles;
		Deque<AttackData> oppProjectiles;
		if(this.playerNumber) {
			myProjectiles = this.frameData.getProjectilesByP1();
			oppProjectiles = this.frameData.getProjectilesByP2();
		} else {
			myProjectiles = this.frameData.getProjectilesByP2();
			oppProjectiles = this.frameData.getProjectilesByP1();
		}
		
		
		int maxProjectiles = 2;
		this.addProjectileInfo(observation, myProjectiles, my, maxProjectiles);
		this.addProjectileInfo(observation, oppProjectiles, opp, maxProjectiles);
		
		
		//Feature selection using specific indices
		List<Integer> indices = Arrays.asList(10, 13, 25, 29, 30, 65, 76, 86, 88, 89, 90, 91, 95, 101, 127, 128, 130, 136);
		Collections.sort(indices, Collections.reverseOrder());
		
		for(int index : indices) {
			if(index >= 0 && index < observation.size()) {
				observation.remove(index);
			}
		}
		
		
		this.clip(observation, LOW_OBSERVATION_SPACE_VALUE, HIGH_OBSERVATION_SPACE_VALUE);
		
		return observation
				.stream()
				.mapToDouble(Double::doubleValue)
				.toArray();
	}
	
	/**
	 * @return the calculated reward
	 */
	private double getReward() {
		if(this.preFrameData.getEmptyFlag() || this.frameData.getEmptyFlag()) return 0.0;
		
		double reward = 0.0, bonus = 0.0;
		
		double p2_hp_pre = this.preFrameData.getCharacter(false).getHp();
        double p1_hp_pre = this.preFrameData.getCharacter(true).getHp();
        double p2_hp_now = this.frameData.getCharacter(false).getHp();
        double p1_hp_now = this.frameData.getCharacter(true).getHp();
        double x_dist_pre = this.preFrameData.getDistanceX();
        double x_dist_now = this.frameData.getDistanceX();
        
        
        if(this.playerNumber) {
        	reward = ((p2_hp_pre-p2_hp_now) - (p1_hp_pre-p1_hp_now)) / 10.0;
        } else {
        	reward = ((p1_hp_pre-p1_hp_now) - (p2_hp_pre-p2_hp_now)) / 10.0;
        }
        
        if(x_dist_now < x_dist_pre) {
        	bonus = +0.01;
        } else if(x_dist_now > x_dist_pre) {
        	bonus = -0.01;
        }
        
        return reward += bonus;
	}
	
	// Utility method: clip values between min and max
	private void clip(List<Double> list, double min, double max) {
        for (int i = 0; i < list.size(); i++) {
            double value = list.get(i);
            if (value < min) {
                list.set(i, min);
            } else if (value > max) {
                list.set(i, max);
            }
        }
    }
	
	// Utility method for one-hot encoding
	private void addOneHotEncodedInfo(List<Double> list, int size, int value) {
    	for(int i = 0; i < size; ++i) {
    		double val = (i == value) ? 1.0 : 0.0;
    		list.add(val);
    	}
    }
	
	/**
	 * Adds projectile information to the observation vector.  
	 * Missing projectiles are zero-padded.
	 *
	 * @param list           observation list
	 * @param projectiles    deque of projectiles
	 * @param ch             character reference
	 * @param maxProjectiles maximum number of projectiles considered
	 */
	private void addProjectileInfo(List<Double> list, Deque<AttackData> projectiles, CharacterData ch, int maxProjectiles) {
		Iterator<AttackData> it = projectiles.iterator();
		int count = 0;
		
		while(it.hasNext() && count < maxProjectiles) {
			AttackData projectile = it.next();
			double myHitDamage = projectile.getHitDamage() / 200.0;
			double myHitAreaNowX = ((projectile.getCurrentHitArea().getLeft() + projectile.getCurrentHitArea().getRight()) / 2.0
		            - (ch.getLeft() + ch.getRight()) / 2.0) / 960.0;
		    double myHitAreaNowY = ((projectile.getCurrentHitArea().getTop() + projectile.getCurrentHitArea().getBottom()) / 2.0) / 640.0;

		    list.add(myHitDamage); 	  // [0 , 1]
		    list.add(myHitAreaNowX); //  [-1, 1]
		    list.add(myHitAreaNowY);//   [0 , 1]
		    count++;
		}
		
		// Zero padding for missing projectiles
		while (count < maxProjectiles) {
		    list.add(0.0);
		    list.add(0.0);
		    list.add(0.0);
		    count++;
		}
	}
}
