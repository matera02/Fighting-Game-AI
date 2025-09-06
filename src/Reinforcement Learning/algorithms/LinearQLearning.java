import java.util.Random;
import java.util.ArrayList;
import java.util.Arrays;
import java.io.*;
/**
 * This class implements Linear Q-Learning.
 */
public class LinearQLearning {
    /** Discount factor */
    private final double gamma = 0.95;
    
    /** Learning rate */
    private double alpha = 0.03;
    
    /** Exploration probability for epsilon-greedy policy */
    private final double epsilon = 0.1;

    /** Number of features per state */
    private final int dim;

    /** Number of possible actions */
    private final int numActions;

    /** Weight vector for linear function approximation */
    private double[] w;
    
    private final Random random = new Random();
    private final ArrayList<Double> rewardData = new ArrayList<>();
    private int numWin = 0;
    
    /** Preallocated matrices and vectors for performance */
    private final double[][] A;
    private final double[][] tiledMatrix;
    private final double[][] normalizedMatrix;
    private final double[] qValues;
    private final double[] x;
    
    /** TD-error */
    private double delta = 0.0;

    /**
     * Constructor for LinearQLearning class.
     * 
     * @param originalFeatureSize number of features in the state
     * @param numActions number of discrete actions available
     */
    public LinearQLearning(int originalFeatureSize, int numActions) {
        this.dim = originalFeatureSize;
        this.numActions = numActions;
        this.w = new double[dim * numActions];
        this.A = createActionMatrix();
        
        // Preallocation
        this.tiledMatrix = new double[numActions][dim];
        this.normalizedMatrix = new double[numActions][dim * numActions];
        this.qValues = new double[numActions];
        this.x = new double[dim * numActions];
    }


    private double[][] createActionMatrix() {
        double[][] matrix = new double[numActions][dim * numActions];
        for (int i = 0; i < numActions; i++) {
            int startIdx = i * dim;
            Arrays.fill(matrix[i], startIdx, startIdx + dim, 1.0);
        }
        return matrix;
    }
    
    
    // Copy the observation values into each row of the tiled matrix
    private void updateTiledMatrix(double[] obs) {
        for (int i = 0; i < numActions; i++) {
            System.arraycopy(obs, 0, tiledMatrix[i], 0, 
            		Math.min(dim, obs.length)); // Ensure we do not exceed the maximum allowed dimension
        }
    }
    
    private void computeQValues() {
        Arrays.fill(qValues, 0.0); // Initialize all Q-values to 0
        
        // Compute Q-values for each action
        for (int i = 0; i < numActions; i++) {
            for (int j = 0; j < normalizedMatrix[0].length; j++) {
                qValues[i] += normalizedMatrix[i][j] * w[j]; // Update each Q-value
            // by summing the product of the normalized matrix elements
            // and their corresponding weights.
            }
        }
    }

    private void normalizeMatrix() {
        for (int i = 0; i < numActions; i++) {
            double rowMax = 0.0;
            // Find the maximum value in the row
            for (int j = 0; j < dim * numActions; j++) {
                rowMax = Math.max(rowMax, Math.abs(tiledMatrix[i][j % dim] * A[i][j]));
            }
            
            if (rowMax > 0) {
                // Normalize the row
                for (int j = 0; j < dim * numActions; j++) {
                    normalizedMatrix[i][j] = (tiledMatrix[i][j % dim] * A[i][j]) / rowMax;
                }
            } else {
            	// Copy the row as is if the maximum value is 0
                for (int j = 0; j < dim * numActions; j++) {
                    normalizedMatrix[i][j] = tiledMatrix[i][j % dim] * A[i][j];
                }
            }
        }
    }

     /**
     * Selects an action using an epsilon-greedy policy.
     * 
     * @param obs current observation
     * @return index of the selected action
     */
    public int selectAction(double[] obs) {
        updateTiledMatrix(obs);
        normalizeMatrix();
        computeQValues();
        
        if (random.nextDouble() < epsilon) {
            return random.nextInt(numActions);
        }
        
        int maxIndex = 0;
        double maxValue = qValues[0];
        for (int i = 1; i < numActions; i++) {
            if (qValues[i] > maxValue) {
                maxValue = qValues[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    /**
     * Updates the weights of the linear function based on observed transition.
     * 
     * @param obs current observation
     * @param action index of the action taken
     * @param reward reward received after taking the action
     * @param newObs new observation after the action
     */
    public void update(double[] obs, int action, double reward, double[] newObs) {
        // Clear and update the feature vector x
        Arrays.fill(x, 0.0);
        if (dim <= obs.length) {
            System.arraycopy(obs, 0, x, dim * action, Math.min(dim, obs.length));
        }

        // Compute Q-values for the next state
        updateTiledMatrix(newObs);
        normalizeMatrix();
        computeQValues();
        
        // Find the maximum Q-value for the next state
        double maxNextQ = qValues[0];
        for (int i = 1; i < qValues.length; i++) {
            maxNextQ = Math.max(maxNextQ, qValues[i]);
        }

        // Compute the current Q-value
        double currentQ = 0;
        for (int i = 0; i < x.length; i++) {
            currentQ += x[i] * w[i];
        }

        // Update the weights: w <- w + alpha * (reward + gamma * maxNextQ - currentQ) * phi(s, a)
        this.delta = reward + gamma * maxNextQ - currentQ;
        for (int i = 0; i < w.length; i++) {
            w[i] += alpha * this.delta * x[i];
        }
    }

    /** --------------
     * Utility methods
     * setter and getter
     * */

    public void updateLearningRate(int iteration) {
        if (iteration == 300) {
            alpha = 0.01;
        }
    }

    public void saveWeightsToFile(String filepath) throws IOException {
        try (DataOutputStream dos = new DataOutputStream(new FileOutputStream(filepath))) {
            // Save the metadata
            dos.writeInt(dim);
            dos.writeInt(numActions);
            
            // Save the weights
            for (double weight : w) {
                dos.writeDouble(weight);
            }
        }
    }

    public void loadWeightsFromFile(String filepath) throws IOException {
        try (DataInputStream dis = new DataInputStream(new FileInputStream(filepath))) {
            // Perform a check on the metadata
            int savedDim = dis.readInt();
            int savedNumActions = dis.readInt();

            if (savedDim != dim || savedNumActions != numActions) {
                throw new IOException("Incompatible weights file: dimensions do not match. " +
                "Expected dim=" + dim + ", numActions=" + numActions +
                ", but found dim=" + savedDim + ", numActions=" + savedNumActions);
            }
            
            // Load the weights
            w = new double[dim * numActions];
            for (int i = 0; i < w.length; i++) {
                w[i] = dis.readDouble();
            }
        }
    }
    
    //like in computeQValues
    public double[] getQValues(double[] state) {
        updateTiledMatrix(state); 
        normalizeMatrix();        
        computeQValues();         
        return Arrays.copyOf(qValues, qValues.length);
    }

    public void addReward(double reward) {
        rewardData.add(reward);
    }

    public void incrementWins() {
        numWin++;
    }

    public int getNumWins() {
        return numWin;
    }

    public ArrayList<Double> getRewardData() {
        return rewardData;
    }
    
    public void setWeights(double[] w) {
    	this.w = w;
    }
    
    public double[] getWeights() {
    	return this.w;
    }
    
    public double getDelta() {
    	return this.delta;
    }
}