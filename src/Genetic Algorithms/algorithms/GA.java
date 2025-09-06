import java.util.List;

/**
 * Abstract class representing a generic Genetic Algorithm (GA).
 *
 * @param <T> the type of individuals in the population
 */
public abstract class GA<T> {
	
    /** Probability for tournament selection */
    private double K = 0.95;

    /** Size of the population */
    private int populationSize;

    /** Mutation probability */
    private double mutationRate;

    /** List containing the current population */
    private List<T> population;
	
    /**
     * Constructor for the GA class.
     *
     * @param populationSize the number of individuals in the population
     * @param mutationRate the probability of mutating each individual
     */
    public GA(int populationSize, double mutationRate) {
        this.populationSize = populationSize;
        this.mutationRate = mutationRate;
    }
	
    /**
     * Initializes the population.
     *
     * @return a list of initial individuals
     */
    protected abstract List<T> initializePopulation();

    /**
     * Selects an individual from the population according to the GA selection strategy.
     *
     * @return the selected individual
     */
    protected abstract T selection();

    /**
     * Performs crossover between two individuals.
     *
     * @param parent1 the first parent
     * @param parent2 the second parent
     * @return an array of offspring generated from the crossover
     */
    protected abstract T[] crossover(T parent1, T parent2);

    /**
     * Applies mutation to an individual.
     *
     * @param individual the individual to mutate
     */
    protected abstract void mutate(T individual);

    /**
     * Executes the complete genetic algorithm.
     */
    protected abstract void run();
	
    /**
     * Returns the current population.
     *
     * @return the list of individuals in the population
     */
    public List<T> getPopulation() {
        return this.population;
    }
	
    /**
     * Sets the current population.
     *
     * @param population the list of individuals to set
     */
    protected void setPopulation(List<T> population) {
        this.population = population;
    }
	
    /**
     * Returns the probability K used in tournament selection.
     *
     * @return the tournament selection probability
     */
    protected double getK() {
        return this.K;
    }
	
    /**
     * Returns the size of the population.
     *
     * @return the population size
     */
    protected int getPopulationSize() {
        return this.populationSize;
    }
	
    /**
     * Returns the mutation probability.
     *
     * @return the mutation rate
     */
    protected double getMutationRate() {
        return this.mutationRate;
    }
}
