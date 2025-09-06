import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Deque;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

import enumerate.Action;
import simulator.Simulator;
import struct.CharacterData;
import struct.FrameData;
import struct.GameData;

/**
 * Class of the Genetic Algorithm used in GAMCTSAI.
 * Each chromosome represents a sequence of possible actions in the game. 
 * The GA evolves these chromosomes to find the sequence of actions 
 * that maximizes fitness based on simulated combat outcomes.
 * */
public class GeneticAlgorithm extends GA<GeneticAlgorithm.Chromosome> {
	
	public static boolean DEBUG_MODE = false;
	public static boolean IS_MY_FITNESS = false;
	
	
	// Genetic algorithm parameters
	private static final int NUM_ACTIONS = 5;
	private int generations;
	private Random random = new Random();
	private Util.TYPE_OF_ACTION typeOfAction;
	
	// Simulation parameters
	private boolean playerNumber;
	private FrameData frameData;
	private GameData gameData;
	private Simulator simulator;
	private Deque<Action> opponentActions;
	private List<Double> fitnessStats;
	
	/**
	 * Constructor for the GeneticAlgorithm class
	 * 
     * @param generations number of generations to run
     * @param populationSize size of the population
     * @param mutationRate probability of mutating each gene
     * @param tp type of action used to encode the chromosome
     * @param frameData current game frame data
     * @param gameData overall game data
     * @param playerNumber identifier for the player
     * @param opponentActions deque of opponent actions for simulation
	 */
	public GeneticAlgorithm(
			int generations, int populationSize, double mutationRate, 
			Util.TYPE_OF_ACTION tp,FrameData frameData, GameData gameData, boolean playerNumber,
			Deque<Action> opponentActions) {
		
		super(populationSize, mutationRate);
		this.generations = generations;
		this.typeOfAction = tp;
		
		this.frameData = frameData;
		this.gameData = gameData;
		
		this.simulator = new Simulator(this.gameData);
		this.playerNumber = playerNumber;
		this.opponentActions = opponentActions;
		
		this.fitnessStats = new ArrayList<>();
		
		this.setPopulation(this.initializePopulation());
	}
	
	/**
     * Initializes the population of chromosomes for the genetic algorithm.
     * 
     * Each chromosome is created with a sequence of actions (genes) that are initialized
     * randomly while respecting the constraint that no duplicate actions are allowed
     * within the same chromosome.
     *
     * @return a list containing the initialized population of chromosomes
     */
	@Override
	protected List<Chromosome> initializePopulation() {
		List<Chromosome> pop = new ArrayList<>();
		for(int i = 0; i < this.getPopulationSize(); ++i) {
			
			if (DEBUG_MODE) {
				System.out.println("TYPE OF ACTION: " + this.typeOfAction);
			}
			
			Chromosome chrm = new Chromosome(GeneticAlgorithm.NUM_ACTIONS, this.typeOfAction, 
												this.frameData, this.opponentActions, 
												this.simulator, this.playerNumber);
			pop.add(chrm);
		}
		return pop;
	}
	
	/**
     * Selects a chromosome from the population using tournament selection.
     *
     * Two chromosomes are randomly chosen from the population. With probability K, the
     * chromosome with higher fitness is selected; otherwise, a chromosome is selected
     * randomly among the two.
     *
     * @return the selected chromosome
     */
	@Override
	protected Chromosome selection() {
		List<Chromosome> tournament = new ArrayList<>();
        tournament.add(this.getPopulation().get(random.nextInt(this.getPopulationSize())));
        tournament.add(this.getPopulation().get(random.nextInt(this.getPopulationSize())));

        if (random.nextDouble() < this.getK()) {
            return Collections.max(tournament, Comparator.comparingDouble(Chromosome::getFitness));
        }
        
        return tournament.get(random.nextInt(2));
	}

	/**
     * Performs a two-point crossover between two parent chromosomes to produce offspring.
	 * 
     * The crossover exchanges genes at two positions between the parents, only if the swap
     * does not introduce duplicate actions in the resulting chromosomes. If no valid swap
     * positions are found, the original parents are returned as offspring.
     *
     * @param parent1 the first parent chromosome
     * @param parent2 the second parent chromosome
     * @return an array containing the two offspring chromosomes resulting from the crossover
     */
	@Override
	protected Chromosome[] crossover(Chromosome parent1, Chromosome parent2) {
		List<Action> actions1 = new ArrayList<>(parent1.getGenes());
	    List<Action> actions2 = new ArrayList<>(parent2.getGenes());
	    List<Integer> crossoverPoints = new ArrayList<>();
	    
	    for (int i = 0; i < actions1.size() && crossoverPoints.size() < 2; i++) {
	        Action action1 = actions1.get(i);
	        Action action2 = actions2.get(i);
	        
	        // Check if swapping at this position would create duplicates
    		// Verify if actionX is already present in actionsY excluding position i
	        boolean canCreateDuplicate1 = actions1.subList(0, i).contains(action2) || actions1.subList(i + 1, actions1.size()).contains(action2);
	        boolean canCreateDuplicate2 = actions2.subList(0, i).contains(action1) || actions2.subList(i + 1, actions2.size()).contains(action1);
	        
	        
	        // If the swap does not create duplicates, add the i-th position
	        if (!canCreateDuplicate1 && !canCreateDuplicate2) {
	            crossoverPoints.add(i);
	        }
	    }
	    
	    // Perform the crossover at the found positions
	    for (int point : crossoverPoints) {
	        Action temp = actions1.get(point);
	        actions1.set(point, actions2.get(point));
	        actions2.set(point, temp);
	    }
	    
	    // Create the new chromosomes
	    Chromosome chrm1 = new Chromosome(GeneticAlgorithm.NUM_ACTIONS, 
	                                    this.typeOfAction, 
	                                    this.frameData, 
	                                    this.opponentActions, 
	                                    this.simulator, 
	                                    this.playerNumber);
	    
	    
	    Chromosome chrm2 = new Chromosome(GeneticAlgorithm.NUM_ACTIONS, 
	                                    this.typeOfAction, 
	                                    this.frameData, 
	                                    this.opponentActions, 
	                                    this.simulator, 
	                                    this.playerNumber);
	    
	    chrm1.setGenes(actions1);
	    chrm2.setGenes(actions2);
	    
	    return new Chromosome[]{chrm1, chrm2};
	}

	/**
     * Mutates a given chromosome with a probability equal to the configured mutation rate.
     * 
     * Mutation consists of replacing a random gene in the chromosome with a new action,
     * chosen randomly according to the type of action used.
     *
     * @param individual the chromosome to mutate
     */
	@Override
	protected void mutate(Chromosome individual) {
		List<Action> actions = null;
		if (random.nextDouble() < this.getMutationRate()) {
            actions = new ArrayList<>(individual.getGenes());
            int position = random.nextInt(actions.size());
            Action action = Util.getRandomAction(typeOfAction);
            actions.set(position, action); //no duplicate check is performed
        }
		if(actions != null) {
			individual.setGenes(actions);
		}
	}
	
	/**
     * Runs the genetic algorithm for the configured number of generations.
     * Evolves the population using selection, crossover, and mutation operators.
     */
	public void run() {
		// Implementation of GA loop
        for (int generation = 0; generation < this.generations; generation++) {

        	if (DEBUG_MODE) {
        		System.out.println("Generazione " + generation);
                // Evaluate the current population
            	int crossNum = 0;
            	for (Chromosome chromosome : this.getPopulation()) {
                    System.out.println("Fitness Cromosoma " + crossNum + ":" + chromosome.getFitness());
                    ++crossNum;
                }
        	}

            List<Chromosome> newPopulation = new ArrayList<>();
            while (newPopulation.size() < this.getPopulationSize()) {
                Chromosome parent1 = this.selection();
                Chromosome parent2 = this.selection();
                Chromosome[] children = this.crossover(parent1, parent2);
                this.mutate(children[0]);
                this.mutate(children[1]);
                newPopulation.add(children[0]);
                newPopulation.add(children[1]);
            }
            
            // Fitness calculation can be skipped here
        	// and performed only when we need the best chromosome for MCTS
            
            // Get the best fitness of the current generation for statistics
            double bestFitness = Collections.max(this.getPopulation(), 
            		Comparator.comparing(Chromosome::getFitness))
            		.getFitness();
            
            this.fitnessStats.add(bestFitness);
            this.setPopulation(newPopulation);
        }
    }
	
	/**
     * Returns the sequence of actions of the best chromosome in the current population.
     *
     * @return deque of actions representing the best chromosome
     */
	public Deque<Action> getBestActions(){
		return Collections.max(this.getPopulation(), 
				Comparator.comparing(Chromosome::getFitness))
				.getGenes();
	}
	
	/**
     * Returns the list of best fitness values tracked across generations.
     *
     * @return list of fitness values
     */
	public List<Double> getFitnessStats() {
		return this.fitnessStats;
	}
	
	/**
     * Inner class representing a chromosome in the genetic algorithm.
     *
     * Each chromosome contains a sequence of actions (genes) and is associated
     * with a fitness value calculated via simulated combat.
     */
	protected class Chromosome implements Comparable<Chromosome> {
		
		private static final int SIMULATION_TIME = 60;
		
		private int size;
		
		private FrameData frameData;
		private Deque<Action> genes;
		private Deque<Action> opponentActions;
		private Simulator simulator;
		private boolean playerNumber;

		/**
         * Constructor for the Chromosome class.
         *
         * @param size number of genes in the chromosome
         * @param tp type of action used to initialize genes
         * @param fd current frame data for simulation
         * @param oppAct opponent actions deque
         * @param sim simulator instance
         * @param playerNumber identifier of the player
         */
		public Chromosome(int size, Util.TYPE_OF_ACTION tp, 
				FrameData fd, Deque<Action> oppAct, 
				Simulator sim, boolean playerNumber) {
			this.size = size;
			this.genes = new LinkedList<>();
			this.frameData = fd;
			this.opponentActions = oppAct;
			this.simulator = sim;
			this.playerNumber = playerNumber;
			this.initializeGenes(tp);
		}
		
		/**
         * Initializes the genes of the chromosome randomly while avoiding duplicates.
         *
         * @param tp type of action for gene initialization
         */
		private void initializeGenes(Util.TYPE_OF_ACTION tp) {
			this.genes.clear();
			if(!tp.equals(Util.TYPE_OF_ACTION.SKILL)) {
				for(int i = 0; i < this.size;) {
					Action act = Util.getRandomAction(tp);
					if(!this.genes.contains(act)) {
						this.genes.add(act);
						++i;
					}
				}
			} else {
				this.genes.add(Action.STAND_D_DF_FC);
				tp = Util.TYPE_OF_ACTION.GROUND;
				for(int i = 0; i < this.size - 1; ) {
					Action act = Util.getRandomAction(tp);
					if(!this.genes.contains(act)) {
						this.genes.add(act);
						++i;
					}
				}
			}
		}
		
		
		@Override
		public int compareTo(Chromosome other) {
			return Double.compare(this.getFitness(), other.getFitness());
		}
		
		public Deque<Action> getGenes() {
			return this.genes;
		}
		
		public void setGenes(List<Action> newGenes) {
			this.genes = new LinkedList<>(newGenes);
		}
		
		/**
         * Computes and returns the fitness of this chromosome.
         *
         * Fitness is calculated based on the result of simulating the game
         * using the chromosome's sequence of actions. 
		 * Two alternative fitness formulas can be used, depending on IS_MY_FITNESS.
         *
         * @return fitness value
         */
		public double getFitness() {
			FrameData fd = this.getSimulationResult();
			return (!GeneticAlgorithm.IS_MY_FITNESS) ? this.getPaperFitness(fd) : this.getMyFitness(fd);
		}
		
		/**
         * Computes the fitness according to related work.
         *
         * @param fd result of the simulation
         * @return fitness value
         */
		private double getPaperFitness(FrameData fd) {
			return fd.getCharacter(this.playerNumber).getHp() - fd.getCharacter(!this.playerNumber).getHp();
		}
		
		/**
         * Computes the proposed custom fitness.
         *
         * @param fd result of the simulation
         * @return fitness value
         */
		private double getMyFitness(FrameData fd) {
			CharacterData player = fd.getCharacter(this.playerNumber);
			CharacterData opponent = fd.getCharacter(!this.playerNumber);
			
			
			double hpDiff =  player.getHp() - opponent.getHp();
			double distancePenalty = -(player.getCenterX() - opponent.getCenterX());
			double hitBonus = player.getHitCount() * 10;			
			
			return 2 * hpDiff + 1.5 * distancePenalty + hitBonus; //i coefficienti sono stati decisi in maniera empirica
		}
		
		/**
         * Simulates the game using this chromosome's actions and a subset of opponent actions.
         *
         * @return frame data resulting from the simulation
         */
		@SuppressWarnings("unchecked")
		private FrameData getSimulationResult() {
			Deque<Action> myActions = this.getGenes();
			Deque<Action> oppActions = new LinkedList<>();
			Random rnd = new Random();
			for(int i = 0; i < 5; ++i) {
				oppActions.add(((List<Action>) this.opponentActions).get(rnd.nextInt(opponentActions.size())));
			}
			
			return this.simulator.simulate(this.frameData, this.playerNumber, myActions, oppActions, Chromosome.SIMULATION_TIME);
		}
	}
}
