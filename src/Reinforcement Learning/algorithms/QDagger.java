import java.util.Deque;
import java.util.LinkedList;

/**
 * This class implements a customized variant of the QDagger algorithm.
 * 
 * This class maintains a student policy based on Linear Q-Learning and 
 * a teacher policy that provides expert Q-values. During learning, the student 
 * updates its weights through both standard Q-learning updates from environment 
 * experience and distillation from the teacher's Q-values.
 * 
 * The algorithm uses Mean Squared Error between student and teacher Q-value 
 * vectors to perform policy distillation.
 * 
 * @param <T> the type of state representation used by the teacher policy
 */
public class QDagger<T> {
	 /** The maximum size of the expert data buffer */
    private static final int BUFFER_SIZE = 3;

	/** The student policy */
    private LinearQLearning student;

	/** The teacher policy */
    private TeacherPolicy<T> teacher;

	/** The current distillation coefficient */
    private double lambdaT;

	/** The decay rate for the distillation coefficient */
    private double lambdaDecay;

	/** Buffer storing expert state-QValues pairs */
    private Deque<ExpertData> expertData;
	
	/** The number of possible actions */
    private int actionSize;
    
	/**
	 * Constructor for the QDagger class.
	 * 
     * @param student student policy
     * @param teacher teacher policy
     * @param lambdaInit initial distillation coefficient
     * @param lambdaDecay decay rate for the distillation coefficient
     * @param actionSize number of possible actions
	 */
    public QDagger(LinearQLearning student, QDagger.TeacherPolicy<T> teacher, 
                  double lambdaInit, double lambdaDecay, int actionSize) {
        this.student = student;
        this.teacher = teacher;
        this.lambdaT = lambdaInit;
        this.lambdaDecay = lambdaDecay;
        this.expertData = new LinkedList<>();
        this.actionSize = actionSize;
    }
    
	/**
     * Collects expert data for the current state and adds it to the buffer.
     * If the buffer is full, the oldest entry is removed (FIFO).
     *
     * @param fd state representation for the teacher
     * @param state state observation vector
     */
    private void collectExpertData(T fd, double[] state) {
        double[] expertQValues = this.teacher.getQValues(fd);
        
        if(this.expertData.size() >= BUFFER_SIZE) {
            this.expertData.pollFirst(); // FIFO removal
        }
        
        this.expertData.add(new ExpertData(state, expertQValues));
    }
    
	/**
     * Computes the distillation loss between student and expert Q-values
     * using Mean Squared Error.
     *
     * @param state current state observation
     * @param expertQValues expert Q-values for the state
     * @return MSE distillation loss
     */
    @SuppressWarnings("unused")
	private double getDistillationLoss(double[] state, double[] expertQValues) {
        double[] studentQValues = student.getQValues(state);
        double sumSquaredDiff = 0.0;
        
        for (int i = 0; i < actionSize; i++) {
            double diff = studentQValues[i] - expertQValues[i];
            sumSquaredDiff += diff * diff;
        }
        
        return sumSquaredDiff / actionSize;
    }
    
	/**
     * Updates the student weights using distillation loss gradient.
     * Applies the MSE gradient to align student Q-values with expert Q-values.
     *
     * @param state the current state
     * @param expertQValues the expert Q-values for the state
     */
    private void updateWeights(double[] state, double[] expertQValues) {
        double[] studentQValues = student.getQValues(state);
        double[] currentWeights = student.getWeights();
        int stateLength = state.length;
        
        for (int action = 0; action < actionSize; action++) {
            int offset = action * stateLength;
            double qDiff = studentQValues[action] - expertQValues[action];
            
            for (int i = 0; i < stateLength; i++) {
                double gradient = 2.0 * qDiff * state[i] / actionSize; // MSE gradient
                currentWeights[offset + i] -= lambdaT * gradient;
            }
        }
        
        student.setWeights(currentWeights);
    }
    
	/**
     * Performs the Dagger update step using all expert data in the buffer.
     * Updates student weights for each expert demonstration and decays
     * the distillation coefficient.
     */
    public void daggerUpdate() {
        for (ExpertData d : this.expertData) {
            double[] state = d.getState();
            double[] expertQValues = d.getQValues();
            
            updateWeights(state, expertQValues);
        }
        
        lambdaT = Math.max(lambdaT * lambdaDecay, 1e-3);
    }
    
	/**
     * Performs a complete QDagger update step.
     * 
	 * This includes:
     * 		1. Standard Q-learning update (TD loss)
     * 		2. Expert data collection
     * 		3. Dagger update
     *
     * @param fd state representation for the teacher
     * @param state current state observation
     * @param action action taken by the student
     * @param reward the reward received
     * @param nextState next state observation
     */
    public void update(T fd, double[] state, int action, double reward, double[] nextState) {
        student.update(state, action, reward, nextState);
        collectExpertData(fd, state);
        daggerUpdate();
    }
    
	/**
     * Functional interface for teacher policy.
     *
     * @param <T> type of state representation used by the teacher policy
     */
    @FunctionalInterface
    public interface TeacherPolicy<T> {
		
		/**
         * @param state the state representation
         * @return an array of Q-values for all actions in the given state.
         */
        double[] getQValues(T state);
    
	}
    
	//** Inner class for storing state and corresponding expert Q-values */
    private class ExpertData {
        private double[] state;
        private double[] qValues;
        
        private ExpertData(double[] state, double[] qValues) {
            this.state = state;
            this.qValues = qValues;
        }
        
        private double[] getState() {
            return this.state;
        }
        
        private double[] getQValues() {
            return this.qValues;
        }
    }
}