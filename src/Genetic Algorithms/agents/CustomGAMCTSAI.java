import java.util.LinkedList;
import java.util.List;

import enumerate.Action;

/**
 * CustomGAMCTSAI extends GAMCTSAI by overriding only 
 * the processing and roundEnd methods.
 *   
 * This class is used for experiments with our custom fitness function and parameters.  
 * Apart from these changes and the destination folders for storing game statistics, 
 * the logic remains the same as in the original GAMCTSAI implementation.
 */
public class CustomGAMCTSAI extends GAMCTSAI {
    
    @Override
    public void processing() {
        this.timer.updateTimer();
        
        if (frameData.getEmptyFlag() || frameData.getFramesNumber() <= 0) {
            return;
        }
        
        if (commandCenter.getSkillFlag()) {
            key = commandCenter.getSkillKey();
        } else {
            key.empty();
            commandCenter.skillCancel();
            
            Util.TYPE_OF_ACTION tp = this.prepare();
            
            GeneticAlgorithm.IS_MY_FITNESS = true;
            GeneticAlgorithm ga = new GeneticAlgorithm(7, 10, 0.005, tp, this.frameData, this.gameData, this.playerNumber, this.oppActions); //my params

			ga.run();
            this.myActions = ga.getBestActions();
            this.fitnessStats.add(ga.getFitnessStats());
            
            this.prepare();
            
            this.rootNode = new MCTSNode(this.simulatorAheadFrameData, null, 
					(LinkedList<Action>) this.myActions, (LinkedList<Action>) this.oppActions, 
					this.gameData, this.playerNumber, this.commandCenter);
            
            this.rootNode.expandNode();
            Action bestAction = this.rootNode.executeMCTS();
            
            if (DEBUG_ENABLED) {
                rootNode.printNodeStructure(rootNode);
            }
            
            commandCenter.commandCall(bestAction.name());
        }
    }
    
    @Override
    public void roundEnd(int p1Hp, int p2Hp, int frames) {
        System.out.println(p1Hp + " " + p2Hp + " " + frames);
        
        List<Integer> hp = List.of(p1Hp, p2Hp, this.timer.getTimerSeconds());
        
        String statsFolder = "custom_stats";
        String fitnessFileName = "custom_fitness_parametri_paper.csv";
        String hpFileName = "custom_hp_parametri_paper.csv";
        
        Util.writeCsv(this.fitnessStats, statsFolder, fitnessFileName, true);
        Util.writeCsv(hp, statsFolder, hpFileName, true);
        
        this.fitnessStats.clear();
        this.timer.resetTimer();
    }
}
