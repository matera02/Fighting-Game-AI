import java.io.IOException;

import struct.FrameData;
import struct.GameData;

/**
 * GARLAgent implements the EvoRL approach.
 *
 * If we want to train the agent for a fixed number of generations, we must take into account
 * the total number of episodes per generation to configure the total number of rounds at startup
 * as NUM_GENERATIONS * EPISODES_PER_GENERATION.
 * 
 * This value should be passed to the run_ftg.sh script when launching the training,
 * while selecting GARLAgent as the active agent.
 */
public class GARLAgent extends BaseRLAgent {
	
	private static final String WEIGHTS_FOLDER = "pesi/geneticRL/";
	private static final int POPULATION_SIZE = 10;
	private static final int TOURNAMENT_SIZE = 3;
	private static final double MUTATION_RATE = 0.01;
	private static final int EPISODES_PER_INDIVIDUAL = 5;
	
	
    /** Total number of episodes for a full generation */
    private static final int EPISODES_PER_GENERATION = POPULATION_SIZE * EPISODES_PER_INDIVIDUAL;
    
    private GeneticWeightsLearner geneticLearner;
    
    private int currentGeneration = 0;
    private int currentIndividualEpisodes;
    private int populationIndex;
    
    /**
     * Initializes the agent by calling the superclass initialization, 
     * initializing the genetic algorithm and setting the weights for
     * the first individual.
	 * 
	 * @param gd GameData object containing the game’s global information
     * @param playerNumber ID of the controlled player (true = P1, false = P2)
     * @return 0 
	 */
    @Override
	public int initialize(GameData gd, boolean playerNumber) {
    	super.initialize(gd, playerNumber);
    	
    	// Initialize the genetic algorithm
        this.geneticLearner = new GeneticWeightsLearner(POPULATION_SIZE,
        	TOURNAMENT_SIZE,
            Environment.OBSERVATION_SPACE_SIZE,
            Environment.ACTION_SPACE_SIZE,
            MUTATION_RATE
        );
        
        // set the weights for the first individual with the best among the candidates 
        // in the initial population, which is random
        this.getLinearQLearning().setWeights(geneticLearner.getPopulation().get(0).getWeights().clone());
        
        this.currentIndividualEpisodes = 0;
        this.populationIndex = 0;
    	return 0;
    }
    
    /**
     * Called at the end of each round.
     * 
     * Handles the evolutionary logic.
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
        
        String dir = "log";
        String fileName = "genetic_rl_results.csv";
        
        Util.logGARLData(dir, fileName, p1Hp, p2Hp, frames, this.currentGeneration, this.populationIndex, this.currentIndividualEpisodes, this.getTotalReward());
        
        currentIndividualEpisodes++;
        int totalEpisodes = this.getRoundNum();
        
         // If the current individual has completed its assigned number of episodes
        if (currentIndividualEpisodes >= EPISODES_PER_INDIVIDUAL) {
        	
            // Set the fitness score for the current individual
            geneticLearner.getPopulation().get(populationIndex)
                .setFitness(this.getTotalReward() / currentIndividualEpisodes);
            
            // Update the individual's weights with those of the trained agent
            geneticLearner.getPopulation()
            	.get(populationIndex)
            	.setWeights(this.getLinearQLearning().getWeights());
            
            populationIndex++;
            currentIndividualEpisodes = 0;
            this.setTotalReward(0.0); // Reset total reward before switching to the next individual
            
            // If all individuals have been evaluated
            if (populationIndex >= POPULATION_SIZE) {
                geneticLearner.run(); // called for evolving population
                populationIndex = 0;
                currentGeneration++;
                
                // Save the best weights every full generation 
                if (totalEpisodes % EPISODES_PER_GENERATION == 0) {
                    try {
                    	this.getLinearQLearning().saveWeightsToFile(WEIGHTS_FOLDER + "best_weights_gen" + currentGeneration + "_ep" + totalEpisodes + ".dat");
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }
            
            // Update the agent with the weights of the next individual
            // (Go to next individual)
            this.getLinearQLearning().setWeights(geneticLearner.getPopulation()
            		.get(populationIndex).getWeights().clone());
        }
        this.reset();
    }

    /**
     * Resets the agent’s internal state for a new round.
     */
    @Override
    protected void reset() {
        // The total reward is not reset here, since it is cleared only when switching to a new individual
        this.getEnv().close();
		this.setReset(false);
		this.setFrameData(new FrameData());

		this.setObservation(new double[Environment.OBSERVATION_SPACE_SIZE]);
		this.setTimeSteps(0);
		this.setRoundNum(this.getRoundNum() + 1);
	}    
}
import java.io.IOException;

import struct.FrameData;
import struct.GameData;

/**
 * GARLAgent implements the EvoRL approach.
 *
 * If we want to train the agent for a fixed number of generations, we must take into account
 * the total number of episodes per generation to configure the total number of rounds at startup
 * as NUM_GENERATIONS * EPISODES_PER_GENERATION.
 * 
 * This value should be passed to the run_ftg.sh script when launching the training,
 * while selecting GARLAgent as the active agent.
 */
public class GARLAgent extends BaseRLAgent {
	
	private static final String WEIGHTS_FOLDER = "pesi/geneticRL/";
	private static final int POPULATION_SIZE = 10;
	private static final int TOURNAMENT_SIZE = 3;
	private static final double MUTATION_RATE = 0.01;
	private static final int EPISODES_PER_INDIVIDUAL = 5;
	
	
    /** Total number of episodes for a full generation */
    private static final int EPISODES_PER_GENERATION = POPULATION_SIZE * EPISODES_PER_INDIVIDUAL;
    
    private GeneticWeightsLearner geneticLearner;
    
    private int currentGeneration = 0;
    private int currentIndividualEpisodes;
    private int populationIndex;
    
    /**
     * Initializes the agent by calling the superclass initialization, 
     * initializing the genetic algorithm and setting the weights for
     * the first individual.
	 * 
	 * @param gd GameData object containing the game’s global information
     * @param playerNumber ID of the controlled player (true = P1, false = P2)
     * @return 0 
	 */
    @Override
	public int initialize(GameData gd, boolean playerNumber) {
    	super.initialize(gd, playerNumber);
    	
    	// Initialize the genetic algorithm
        this.geneticLearner = new GeneticWeightsLearner(POPULATION_SIZE,
        	TOURNAMENT_SIZE,
            Environment.OBSERVATION_SPACE_SIZE,
            Environment.ACTION_SPACE_SIZE,
            MUTATION_RATE
        );
        
        // set the weights for the first individual with the best among the candidates 
        // in the initial population, which is random
        this.getLinearQLearning().setWeights(geneticLearner.getPopulation().get(0).getWeights().clone());
        
        this.currentIndividualEpisodes = 0;
        this.populationIndex = 0;
    	return 0;
    }
    
    /**
     * Called at the end of each round.
     * 
     * Handles the evolutionary logic.
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
        
        String dir = "log";
        String fileName = "genetic_rl_results.csv";
        
        Util.logGARLData(dir, fileName, p1Hp, p2Hp, frames, this.currentGeneration, this.populationIndex, this.currentIndividualEpisodes, this.getTotalReward());
        
        currentIndividualEpisodes++;
        int totalEpisodes = this.getRoundNum();
        
         // If the current individual has completed its assigned number of episodes
        if (currentIndividualEpisodes >= EPISODES_PER_INDIVIDUAL) {
        	
            // Set the fitness score for the current individual
            geneticLearner.getPopulation().get(populationIndex)
                .setFitness(this.getTotalReward() / currentIndividualEpisodes);
            
            // Update the individual's weights with those of the trained agent
            geneticLearner.getPopulation()
            	.get(populationIndex)
            	.setWeights(this.getLinearQLearning().getWeights());
            
            populationIndex++;
            currentIndividualEpisodes = 0;
            this.setTotalReward(0.0); // Reset total reward before switching to the next individual
            
            // If all individuals have been evaluated
            if (populationIndex >= POPULATION_SIZE) {
                geneticLearner.run(); // called for evolving population
                populationIndex = 0;
                currentGeneration++;
                
                // Save the best weights every full generation 
                if (totalEpisodes % EPISODES_PER_GENERATION == 0) {
                    try {
                    	this.getLinearQLearning().saveWeightsToFile(WEIGHTS_FOLDER + "best_weights_gen" + currentGeneration + "_ep" + totalEpisodes + ".dat");
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }
            
            // Update the agent with the weights of the next individual
            // (Go to next individual)
            this.getLinearQLearning().setWeights(geneticLearner.getPopulation()
            		.get(populationIndex).getWeights().clone());
        }
        this.reset();
    }

    /**
     * Resets the agent’s internal state for a new round.
     */
    @Override
    protected void reset() {
        // The total reward is not reset here, since it is cleared only when switching to a new individual
        this.getEnv().close();
		this.setReset(false);
		this.setFrameData(new FrameData());

		this.setObservation(new double[Environment.OBSERVATION_SPACE_SIZE]);
		this.setTimeSteps(0);
		this.setRoundNum(this.getRoundNum() + 1);
	}    
}
import java.io.IOException;

import struct.FrameData;
import struct.GameData;

/**
 * GARLAgent implements the EvoRL approach.
 *
 * If we want to train the agent for a fixed number of generations, we must take into account
 * the total number of episodes per generation to configure the total number of rounds at startup
 * as NUM_GENERATIONS * EPISODES_PER_GENERATION.
 * 
 * This value should be passed to the run_ftg.sh script when launching the training,
 * while selecting GARLAgent as the active agent.
 */
public class GARLAgent extends BaseRLAgent {
	
	private static final String WEIGHTS_FOLDER = "pesi/geneticRL/";
	private static final int POPULATION_SIZE = 10;
	private static final int TOURNAMENT_SIZE = 3;
	private static final double MUTATION_RATE = 0.01;
	private static final int EPISODES_PER_INDIVIDUAL = 5;
	
	
    /** Total number of episodes for a full generation */
    private static final int EPISODES_PER_GENERATION = POPULATION_SIZE * EPISODES_PER_INDIVIDUAL;
    
    private GeneticWeightsLearner geneticLearner;
    
    private int currentGeneration = 0;
    private int currentIndividualEpisodes;
    private int populationIndex;
    
    /**
     * Initializes the agent by calling the superclass initialization, 
     * initializing the genetic algorithm and setting the weights for
     * the first individual.
	 * 
	 * @param gd GameData object containing the game’s global information
     * @param playerNumber ID of the controlled player (true = P1, false = P2)
     * @return 0 
	 */
    @Override
	public int initialize(GameData gd, boolean playerNumber) {
    	super.initialize(gd, playerNumber);
    	
    	// Initialize the genetic algorithm
        this.geneticLearner = new GeneticWeightsLearner(POPULATION_SIZE,
        	TOURNAMENT_SIZE,
            Environment.OBSERVATION_SPACE_SIZE,
            Environment.ACTION_SPACE_SIZE,
            MUTATION_RATE
        );
        
        // set the weights for the first individual with the best among the candidates 
        // in the initial population, which is random
        this.getLinearQLearning().setWeights(geneticLearner.getPopulation().get(0).getWeights().clone());
        
        this.currentIndividualEpisodes = 0;
        this.populationIndex = 0;
    	return 0;
    }
    
    /**
     * Called at the end of each round.
     * 
     * Handles the evolutionary logic.
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
        
        String dir = "log";
        String fileName = "genetic_rl_results.csv";
        
        Util.logGARLData(dir, fileName, p1Hp, p2Hp, frames, this.currentGeneration, this.populationIndex, this.currentIndividualEpisodes, this.getTotalReward());
        
        currentIndividualEpisodes++;
        int totalEpisodes = this.getRoundNum();
        
         // If the current individual has completed its assigned number of episodes
        if (currentIndividualEpisodes >= EPISODES_PER_INDIVIDUAL) {
        	
            // Set the fitness score for the current individual
            geneticLearner.getPopulation().get(populationIndex)
                .setFitness(this.getTotalReward() / currentIndividualEpisodes);
            
            // Update the individual's weights with those of the trained agent
            geneticLearner.getPopulation()
            	.get(populationIndex)
            	.setWeights(this.getLinearQLearning().getWeights());
            
            populationIndex++;
            currentIndividualEpisodes = 0;
            this.setTotalReward(0.0); // Reset total reward before switching to the next individual
            
            // If all individuals have been evaluated
            if (populationIndex >= POPULATION_SIZE) {
                geneticLearner.run(); // called for evolving population
                populationIndex = 0;
                currentGeneration++;
                
                // Save the best weights every full generation 
                if (totalEpisodes % EPISODES_PER_GENERATION == 0) {
                    try {
                    	this.getLinearQLearning().saveWeightsToFile(WEIGHTS_FOLDER + "best_weights_gen" + currentGeneration + "_ep" + totalEpisodes + ".dat");
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }
            
            // Update the agent with the weights of the next individual
            // (Go to next individual)
            this.getLinearQLearning().setWeights(geneticLearner.getPopulation()
            		.get(populationIndex).getWeights().clone());
        }
        this.reset();
    }

    /**
     * Resets the agent’s internal state for a new round.
     */
    @Override
    protected void reset() {
        // The total reward is not reset here, since it is cleared only when switching to a new individual
        this.getEnv().close();
		this.setReset(false);
		this.setFrameData(new FrameData());

		this.setObservation(new double[Environment.OBSERVATION_SPACE_SIZE]);
		this.setTimeSteps(0);
		this.setRoundNum(this.getRoundNum() + 1);
	}    
}
import java.io.IOException;

import struct.FrameData;
import struct.GameData;

/**
 * GARLAgent implements the EvoRL approach.
 *
 * If we want to train the agent for a fixed number of generations, we must take into account
 * the total number of episodes per generation to configure the total number of rounds at startup
 * as NUM_GENERATIONS * EPISODES_PER_GENERATION.
 * 
 * This value should be passed to the run_ftg.sh script when launching the training,
 * while selecting GARLAgent as the active agent.
 */
public class GARLAgent extends BaseRLAgent {
	
	private static final String WEIGHTS_FOLDER = "pesi/geneticRL/";
	private static final int POPULATION_SIZE = 10;
	private static final int TOURNAMENT_SIZE = 3;
	private static final double MUTATION_RATE = 0.01;
	private static final int EPISODES_PER_INDIVIDUAL = 5;
	
	
    /** Total number of episodes for a full generation */
    private static final int EPISODES_PER_GENERATION = POPULATION_SIZE * EPISODES_PER_INDIVIDUAL;
    
    private GeneticWeightsLearner geneticLearner;
    
    private int currentGeneration = 0;
    private int currentIndividualEpisodes;
    private int populationIndex;
    
    /**
     * Initializes the agent by calling the superclass initialization, 
     * initializing the genetic algorithm and setting the weights for
     * the first individual.
	 * 
	 * @param gd GameData object containing the game’s global information
     * @param playerNumber ID of the controlled player (true = P1, false = P2)
     * @return 0 
	 */
    @Override
	public int initialize(GameData gd, boolean playerNumber) {
    	super.initialize(gd, playerNumber);
    	
    	// Initialize the genetic algorithm
        this.geneticLearner = new GeneticWeightsLearner(POPULATION_SIZE,
        	TOURNAMENT_SIZE,
            Environment.OBSERVATION_SPACE_SIZE,
            Environment.ACTION_SPACE_SIZE,
            MUTATION_RATE
        );
        
        // set the weights for the first individual with the best among the candidates 
        // in the initial population, which is random
        this.getLinearQLearning().setWeights(geneticLearner.getPopulation().get(0).getWeights().clone());
        
        this.currentIndividualEpisodes = 0;
        this.populationIndex = 0;
    	return 0;
    }
    
    /**
     * Called at the end of each round.
     * 
     * Handles the evolutionary logic.
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
        
        String dir = "log";
        String fileName = "genetic_rl_results.csv";
        
        Util.logGARLData(dir, fileName, p1Hp, p2Hp, frames, this.currentGeneration, this.populationIndex, this.currentIndividualEpisodes, this.getTotalReward());
        
        currentIndividualEpisodes++;
        int totalEpisodes = this.getRoundNum();
        
         // If the current individual has completed its assigned number of episodes
        if (currentIndividualEpisodes >= EPISODES_PER_INDIVIDUAL) {
        	
            // Set the fitness score for the current individual
            geneticLearner.getPopulation().get(populationIndex)
                .setFitness(this.getTotalReward() / currentIndividualEpisodes);
            
            // Update the individual's weights with those of the trained agent
            geneticLearner.getPopulation()
            	.get(populationIndex)
            	.setWeights(this.getLinearQLearning().getWeights());
            
            populationIndex++;
            currentIndividualEpisodes = 0;
            this.setTotalReward(0.0); // Reset total reward before switching to the next individual
            
            // If all individuals have been evaluated
            if (populationIndex >= POPULATION_SIZE) {
                geneticLearner.run(); // called for evolving population
                populationIndex = 0;
                currentGeneration++;
                
                // Save the best weights every full generation 
                if (totalEpisodes % EPISODES_PER_GENERATION == 0) {
                    try {
                    	this.getLinearQLearning().saveWeightsToFile(WEIGHTS_FOLDER + "best_weights_gen" + currentGeneration + "_ep" + totalEpisodes + ".dat");
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }
            
            // Update the agent with the weights of the next individual
            // (Go to next individual)
            this.getLinearQLearning().setWeights(geneticLearner.getPopulation()
            		.get(populationIndex).getWeights().clone());
        }
        this.reset();
    }

    /**
     * Resets the agent’s internal state for a new round.
     */
    @Override
    protected void reset() {
        // The total reward is not reset here, since it is cleared only when switching to a new individual
        this.getEnv().close();
		this.setReset(false);
		this.setFrameData(new FrameData());

		this.setObservation(new double[Environment.OBSERVATION_SPACE_SIZE]);
		this.setTimeSteps(0);
		this.setRoundNum(this.getRoundNum() + 1);
	}    
}
