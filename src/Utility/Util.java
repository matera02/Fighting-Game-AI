import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import enumerate.Action;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Paths;
import struct.CharacterData;
import struct.MotionData;
import enumerate.State;

/*
 * Utility class. 
 */
public interface Util {

	public static final int FPS = 60; // The game runs at 60 FPS

	public static enum TYPE_OF_ACTION {
		AIR, GROUND, SKILL
	}

	static final Random random = new Random();

	public static Action getRandomAction(TYPE_OF_ACTION typeOfAction) {
		List<Action> actions = Util.getActions(typeOfAction);
		int actionIndex = random.nextInt(0, actions.size());
		return actions.get(actionIndex);
	}

	public static List<Action> getActions(TYPE_OF_ACTION typeOfAction) {
		List<Action> actions = new LinkedList<>();

		switch (typeOfAction) {
		case AIR:
			actions = new LinkedList<>(
					List.of(Action.AIR_GUARD, Action.AIR_A, Action.AIR_B, Action.AIR_DA, Action.AIR_DB, Action.AIR_FA,
							Action.AIR_FB, Action.AIR_UA, Action.AIR_UB, Action.AIR_D_DF_FA, Action.AIR_D_DF_FB,
							Action.AIR_F_D_DFA, Action.AIR_F_D_DFB, Action.AIR_D_DB_BA, Action.AIR_D_DB_BB));
			break;

		case GROUND:
			actions = new LinkedList<>(List.of(Action.STAND_D_DB_BA, Action.BACK_STEP, Action.FORWARD_WALK, Action.DASH,
					Action.JUMP, Action.FOR_JUMP, Action.BACK_JUMP, Action.STAND_GUARD, Action.CROUCH_GUARD,
					Action.THROW_A, Action.THROW_B, Action.STAND_A, Action.STAND_B, Action.CROUCH_A, Action.CROUCH_B,
					Action.STAND_FA, Action.STAND_FB, Action.CROUCH_FA, Action.CROUCH_FB, Action.STAND_D_DF_FA,
					Action.STAND_D_DF_FB, Action.STAND_F_D_DFA, Action.STAND_F_D_DFB, Action.STAND_D_DB_BB));
			break;

		case SKILL:
			actions = new LinkedList<>(List.of(Action.STAND_D_DF_FC));
			break;

		default:
			break;

		}

		return actions;
	}

	/**
	 * Returns the legal actions for a character given the current frame state.
	 *
	 * @param character    the character
	 * @param motionData   the motion data of the character
	 * @param spSkill      the special move
	 * @param actionAir    list of air actions
	 * @param actionGround list of ground actions
	 * @return list of legal actions
	 */
	public static List<Action> getLegalActions(CharacterData character, List<MotionData> motionData, Action spSkill,
			List<Action> airActions, List<Action> groundActions) {
		List<Action> legalActions = new ArrayList<>();
		int energy = character.getEnergy();
		if (character.getState() == State.AIR) {
			for (Action action : airActions) {
				if (isActionExecutable(action, energy, motionData)) {
					legalActions.add(action);
				}
			}
		} else {
			if (isActionExecutable(spSkill, energy, motionData)) {
				legalActions.add(spSkill);
			}
			for (Action action : groundActions) {
				if (isActionExecutable(action, energy, motionData)) {
					legalActions.add(action);
				}
			}
		}
		return legalActions;
	}

	/**
	 * Checks if an action is executable given the available energy.
	 *
	 * @param action     the action to check
	 * @param energy     the available energy
	 * @param motionData the motion data for energy cost lookup
	 * @return true if the action is executable, false otherwise
	 */
	public static boolean isActionExecutable(Action action, int energy, List<MotionData> motionData) {
		try {
			int energyCost = Math.abs(motionData.get(action.ordinal()).getAttackStartAddEnergy());
			return energyCost <= energy;
		} catch (IndexOutOfBoundsException e) {
			return false;
		}
	}

	public static <T> void writeCsv(List<?> data, String folderName, String fileName, boolean append) {
		try {
			String currentPath = Paths.get("").toAbsolutePath().toString(); // Take current working directory
			File folder = new File(currentPath + File.separator + folderName);
			if (!folder.exists()) { // If it doesn’t exist, create a folder
				folder.mkdir();
			}

			File csvFile = new File(folder, fileName);
			try (FileWriter writer = new FileWriter(csvFile, append)) {

				List<?> dataCopy = new ArrayList<>(data); // make a copy to avoid exceptions

				// Check if data is a list of lists or a simple list
				if (!dataCopy.isEmpty()) {
					if (dataCopy.get(0) instanceof List) {
						// Handle a list of lists
						for (Object row : dataCopy) {
							List<?> listRow = (List<?>) row;
							Util.writeCsvRow(writer, listRow);
						}
					} else {
						// Handle a simple list
						Util.writeCsvRow(writer, dataCopy);
					}
				}
			}
			System.out.println("File CSV aggiornato con successo: " + csvFile.getAbsolutePath());
		} catch (IOException e) {
			// e.printStackTrace();
			System.err.println("ERRORE SCRITTURA CSV");
		}
	}

	/**
	 * Helper method to write a single CSV row
	 */
	private static void writeCsvRow(FileWriter writer, List<?> row) throws IOException {
		String csvRow = String.join(",", row.stream().map(String::valueOf).toArray(String[]::new));
		writer.write(csvRow + "\n");
	}

	/** logData for RLAgent during training */
	public static void logRLData(String directoryName, String fileName, int p1Hp, int p2Hp, int frames, int timeSteps,
			int roundNum, double totalReward) {
		try {

			File directory = new File(directoryName);
			if (!directory.exists()) {
				if (directory.mkdir()) {
					System.out.println("Cartella creata: " + directoryName);
				} else {
					System.err.println("Errore nella creazione della cartella: " + directoryName);
					return; // Stop if folder creation failed
				}
			}

			// Full file path
			File file = new File(directoryName + File.separator + fileName);

			// Write data in CSV format
			try (FileWriter writer = new FileWriter(file, true)) {
				writer.append(roundNum + ",");
				writer.append(p1Hp + "," + p2Hp + "," + frames + ",");
				writer.append(totalReward + ",");
				writer.append("" + timeSteps);
				writer.append("\n");
			}

		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	/** logData for TestRLAgent */
	public static void logRLData(String directoryName, String fileName, int p1Hp, int p2Hp, int frames, int timeSteps,
			int roundNum, double totalReward, int timeElapsed) {
		try {
			File directory = new File(directoryName);
			if (!directory.exists()) {
				if (directory.mkdir()) {
					System.out.println("Cartella creata: " + directoryName);
				} else {
					System.err.println("Errore nella creazione della cartella: " + directoryName);
					return;
				}
			}

			// Full file path
			File file = new File(directoryName + File.separator + fileName);

			// Write data in CSV format
			try (FileWriter writer = new FileWriter(file, true)) {
				writer.append(roundNum + ",");
				writer.append(p1Hp + "," + p2Hp + "," + frames + ",");
				writer.append(totalReward + ",");
				writer.append("" + timeSteps + ",");
				writer.append("" + timeElapsed);
				writer.append("\n");
			}

		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	/** log data for GARLAgent */
	public static void logGARLData(String directoryName, String fileName, int p1Hp, int p2Hp, int frames,
			int currentGeneration, int populationIndex, int currentIndividualEpisodes, double totalReward) {
		try {
			File directory = new File(directoryName);
			if (!directory.exists()) {
				if (directory.mkdir()) {
					System.out.println("Directory creata: " + directoryName);
				} else {
					System.err.println("Errore nella creazione della directory: " + directoryName);
					return;
				}
			}

			File file = new File(directoryName + File.separator + fileName);
			boolean isNewFile = !file.exists();
			try (FileWriter writer = new FileWriter(file, true)) {

				// Add Headers
				if (isNewFile) {
					writer.append("Generation,Individual,Episode,P1_HP,P2_HP,Frames,Total_Reward\n");
				}

				writer.append(currentGeneration + ",");
				writer.append((populationIndex + 1) + ",");
				writer.append((currentIndividualEpisodes + 1) + ",");
				writer.append(p1Hp + "," + p2Hp + "," + frames + ",");
				writer.append(totalReward + "\n");

			}

		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static class Timer {

		public static final int TIMER_DELAY = 2;

		private long startTime;
		private int timerSeconds;

		public Timer() {
			this.startTime = System.currentTimeMillis();
			this.timerSeconds = 0;
		}

		public void updateTimer() {
			this.timerSeconds = (int) ((System.currentTimeMillis() - this.startTime) / 1000);
		}

		public void resetTimer() {
			this.startTime = System.currentTimeMillis();
			this.timerSeconds = 0;
		}

		public int getTimerSeconds() {
			return this.timerSeconds - TIMER_DELAY;
		}
	}
}
