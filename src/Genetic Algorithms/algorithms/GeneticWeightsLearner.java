import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

/**
 * Genetic Algorithm used as a learning method for obtaining 
 * the best weight configuration of a Linear Q-Learning based Agent.  
 * 
 * This class is used in GARLAgent.
 */
public class GeneticWeightsLearner extends GA<GeneticWeightsLearner.Individual> {
	
	private final int tournamentSize;
	private final int chromosomeLength;
	private final Random random = new Random();
	
    /**
     * Constructor for the GeneticWeightsLearner class.
     * 
     * @param populationSize size of the population
	 * @param tournamentSize size of the tournament used for selection
	 * @param featureSize dimensionality of the observation space
	 * @param numActions number of possible actions
	 * @param mutationRate probability of mutating each gene
     */
	public GeneticWeightsLearner(int populationSize, int tournamentSize, int featureSize, int numActions, 
            double mutationRate) {
		
		super(populationSize, mutationRate);
		this.tournamentSize = tournamentSize;
		
		this.chromosomeLength = featureSize * numActions;
		
		this.setPopulation(this.initializePopulation());
	}
	
	/**
	 * Initializes the population with small random Gaussian values.
	 * Each individual represents a candidate weight vector.
     * 
     * @return a list containing the initialized population of individuals
	 */
	@Override
	protected List<GeneticWeightsLearner.Individual> initializePopulation() {
		List<Individual> pop = new ArrayList<>();
        for (int i = 0; i < this.getPopulationSize(); i++) {
            double[] weights = new double[chromosomeLength];
            for (int j = 0; j < chromosomeLength; j++) {
                weights[j] = random.nextGaussian() * 0.1;
            }
            pop.add(new Individual(weights));
        }
        return pop;
	}

    /**
	 * Tournament selection with probability k.
	 * Candidates are randomly chosen until the tournament size is reached.
	 * The best candidate is selected with probability k, otherwise a random
	 * candidate from the tournament is returned.
     * 
     * @return the selected individual
	 */
	@Override
	protected GeneticWeightsLearner.Individual selection() {
		List<Individual> tournament = new ArrayList<>();
        
        while(tournament.size() < this.tournamentSize) {
            Individual candidate = this.getPopulation().get(random.nextInt(this.getPopulationSize()));
            if(!tournament.contains(candidate)) {
                tournament.add(candidate);
            }
        }
        
        if(random.nextDouble() < this.getK()) {
            return Collections.max(tournament, Comparator.comparingDouble(Individual::getFitness));
        }
        
        return tournament.get(random.nextInt(tournament.size()));
	}

    /**
     * Performs two-point crossover between two parents to generate offspring.
     * 
     * @param parent1 the first parent individual
     * @param parent2 the second parent individual
     * @return an array containing the two offspring individuals resulting from the crossover
     */
	@Override
	protected GeneticWeightsLearner.Individual[] crossover(GeneticWeightsLearner.Individual parent1,
			GeneticWeightsLearner.Individual parent2) {
        double[] weights1 = parent1.getWeights();
        double[] weights2 = parent2.getWeights();
        
        double[] offspring1 = weights1.clone();
        double[] offspring2 = weights2.clone();
        
        // Two-point crossover
        int point1 = random.nextInt(chromosomeLength);
        int point2 = random.nextInt(chromosomeLength);
        
        if (point1 > point2) {
            int temp = point1;
            point1 = point2;
            point2 = temp;
        }
        
        for (int i = point1; i <= point2; i++) {
            offspring1[i] = weights2[i];
            offspring2[i] = weights1[i];
        }
        
        return new Individual[]{
            new Individual(offspring1),
            new Individual(offspring2)
        };
	}

    /**
	 * Applies Gaussian mutation to the individual's weights.
	 *
	 * @param individual the individual to mutate
	 */
	@Override
	protected void mutate(GeneticWeightsLearner.Individual individual) {
		double[] newWeights = individual.getWeights().clone();
        
        for (int i = 0; i < newWeights.length; i++) {
            if (random.nextDouble() < this.getMutationRate()) {
                newWeights[i] += random.nextGaussian() * 0.1;
            }
        }
        
        individual.setWeights(newWeights);
	}

    /**
     * Executes a single generation of the genetic algorithm.
     * 
     * Unlike standard GA implementations that evolve populations over 
     * multiple generations internally, in this case the evolutionary loop 
     * is externally controlled: multiple calls to this method are triggered 
     * from the roundEnd() method of GARLAgent class.
     */
	@Override
	protected void run() {
		List<Individual> newPopulation = new ArrayList<>();
        
        while(newPopulation.size() < this.getPopulationSize()) {
        	Individual parent1 = this.selection();
        	Individual parent2 = this.selection();
        	Individual[] children = this.crossover(parent1, parent2);
        	this.mutate(children[0]);
        	this.mutate(children[1]);
        	newPopulation.add(children[0]);
        	newPopulation.add(children[1]);
        }
        this.setPopulation(newPopulation);
	}
	
	/**
	 * @return the weight vector of the fittest individual in the population.
	 */
	public double[] getBestWeights() {
    	return Collections.max(this.getPopulation(), 
    			Comparator.comparingDouble(Individual::getFitness)).getWeights();
    }
	
    
	/**
     * Inner class representing an individual in the genetic algorithm.
     * 
     * Each individual consists of a weight vector and its associated fitness.
     */
	protected static class Individual implements Comparable<Individual>{
        private double[] weights;
        private double fitness;

        /**
         * Constructor for the Individual class.
         * The fitness is initialized to negative infinity until evaluation.
         * 
         * @param weights the initial weight vector
         */
        public Individual(double[] weights) {
            this.weights = weights;
            this.fitness = Double.NEGATIVE_INFINITY;
        }
        
        public double[] getWeights() {
            return weights;
        }
        
        public void setWeights(double[] w) {
        	this.weights = w;
        }

        public double getFitness() {
            return fitness;
        }

        public void setFitness(double fitness) {
            this.fitness = fitness;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            Individual that = (Individual) o;
            return Arrays.equals(weights, that.weights) &&
                   Double.compare(fitness, that.fitness) == 0;
        }

        @Override
        public int hashCode() {
            int result = Arrays.hashCode(weights);
            long temp = Double.doubleToLongBits(fitness); 
            result = 31 * result + (int) (temp ^ (temp >>> 32));
            return result;
        }

		@Override
		public int compareTo(Individual o) {
			return Double.compare(this.getFitness(), o.getFitness());
		}
    }
}
