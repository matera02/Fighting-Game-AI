# Dependencies

- **MCTSAgent**
  
  - MCTSNode, Util
    
- **GAMCTSAI**
  
  - GA, GeneticAlgorithm, MCTSNode, Util
    
- **CustomGAMCTSAI**
  
  - GA, GeneticAlgorithm, MCTSNode, Util, GAMCTSAI
    
- **TestEnvironment**
  
  - Environment, Util
    
- **TrainRLAgent**
  
  - BaseRLAgent, Environment, LinearQLearning, Util
    
- **TestRLAgent**
  
  - BaseRLAgent, Environment, LinearQLearning, Util
    
- **SelfPlayAgent**
  
  - MCSTAgent, MCTSNode, BaseRLAgent, Environment, LinearQLearning, TestRLAgent, Util
    
- **GARLAgent**
  
  - BaseRLAgent, Environment, LinearQLearning, GA, GeneticWeightsLearner, Util
    
- **QDaggerAgent**
  
  - BaseRLAgent, Environment, LinearQLearning, QDagger, MCTSNode, Util
