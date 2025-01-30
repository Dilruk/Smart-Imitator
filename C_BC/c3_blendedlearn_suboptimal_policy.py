import copy
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, f1_score, recall_score
import numpy as np
import Utils.a3_utils_updated_division as utils
import tensorflow as tf
from tensorflow.keras import regularizers
from sklearn.utils import shuffle
import pandas as pd
from sklearn.model_selection import train_test_split


def build_policy_net(input_size, output_size):
    # Define a neural network for the policy
    policy_net = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_size,)),
        tf.keras.layers.Dense(int(input_size/2), activation="tanh", kernel_regularizer=regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(int(input_size), activation="relu", kernel_regularizer=regularizers.l2(0.002)),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(int(input_size / 2), activation="relu", kernel_regularizer=regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(int(output_size * 2), activation="relu", kernel_regularizer=regularizers.l2(0.002)),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(output_size, activation="softmax")
    ])

    # Compile the policy network
    # SGD was better than Adam
    print('SGD')
    policy_net.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9), loss="categorical_crossentropy", metrics=['accuracy'])
    return policy_net


def train_policy_net(policy_net, train_data, train_labels, test_data, test_labels, num_epochs=10000, batch_size=1024):
    # Early stopping callback to halt training when validation loss doesn't improve
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

    # Train the policy network using the fit method to take advantage of Keras features like callbacks and validation split
    history = policy_net.fit(
        train_data,
        train_labels,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stop_callback],
        verbose=2  # Set to 2 for one line per epoch, 1 for a progress bar
    )

    # Evaluate the trained model on the test set
    test_loss, test_accuracy = policy_net.evaluate(test_data, test_labels, verbose=0)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Make predictions on the test set
    preds = policy_net.predict(test_data)
    preds_int = np.argmax(preds, axis=1)
    test_labels_int = np.argmax(test_labels, axis=1)

    # Calculate F1 and recall scores
    f1 = f1_score(test_labels_int, preds_int, average='weighted')
    recall = recall_score(test_labels_int, preds_int, average='weighted')
    print(f"F1 Score: {f1:.4f}, Recall: {recall:.4f}")

    # Optionally, return the training history to inspect metrics
    return policy_net, history


def test_policy_net(policy_net, test_data, test_labels):
    with tf.keras.backend.learning_phase_scope(0):
        test_loss = policy_net.evaluate(test_data, test_labels, verbose=0)
        test_accuracy = 0
        for i in range(len(test_data)):
            x = np.array([test_data[i]])
            true_label = np.argmax(test_labels[i])
            pred = policy_net.predict(x)[0]
            pred_label = np.argmax(pred)
            if true_label == pred_label:
                test_accuracy += 1

        test_accuracy /= len(test_data)
        print(f"Loss on test set: {test_loss}")
        print(f"Accuracy on test set: {test_accuracy}")

def compute_action_distribution(train_actions):
    # Count the occurrences of each action
    train_action_counts = np.sum(train_actions, axis=0)
    train_action_percentages = train_action_counts / len(train_actions) * 100
    return train_action_percentages


def handle_data_imbalance(observations, actions):
    import numpy as np
    from imblearn.over_sampling import RandomOverSampler
    from sklearn.preprocessing import LabelBinarizer

    # Assuming observations and actions are already in ndarrays
    # observations: shape (num_samples, 49)
    # actions: shape (num_samples, 25)

    # Convert one-hot encoded action vectors to integer labels
    action_labels = np.argmax(actions, axis=1)

    # Calculate the original class distribution percentages
    unique_classes, class_counts = np.unique(action_labels, return_counts=True)
    total_samples = len(action_labels)
    original_percentages = class_counts / total_samples

    # Determine the target class distribution percentages based on a minimum percentage threshold
    majority_percentage = np.max(original_percentages)
    min_percentage_threshold = majority_percentage / 4

    # Calculate the desired number of samples for each class after applying the oversampling,
    # excluding classes with zero occurrences
    desired_samples_per_class = {
        class_label: int(max(min_percentage_threshold * total_samples, count))
        for class_label, count in zip(unique_classes, class_counts)
        if count > 0
    }

    # Create the RandomOverSampler with the desired sampling strategy
    over_sampler = RandomOverSampler(sampling_strategy=desired_samples_per_class)
    # Apply the oversampling
    oversampled_observations, oversampled_action_labels = over_sampler.fit_resample(observations, action_labels)
    # Shuffle the oversampled dataset
    oversampled_observations, oversampled_action_labels = shuffle(oversampled_observations, oversampled_action_labels, random_state=42)

    # Fit the LabelBinarizer on the integer labels to ensure correct inverse_transform later
    binarizer = LabelBinarizer()
    binarizer.fit(np.arange(25))

    # Convert the balanced integer action labels back to one-hot encoded vectors
    oversampled_actions = binarizer.transform(oversampled_action_labels)

    # Calculate the oversampled class distribution percentages
    oversampled_unique_classes, oversampled_class_counts = np.unique(oversampled_action_labels, return_counts=True)
    oversampled_total_samples = len(oversampled_action_labels)
    oversampled_percentages = oversampled_class_counts / oversampled_total_samples

    # Print the oversampled percentages
    print("Oversampled Class Percentages:")
    for class_label, percentage in zip(oversampled_unique_classes, oversampled_percentages):
        print(f"Class {class_label}: {percentage * 100:.2f}%")
    return oversampled_observations, oversampled_actions, oversampled_action_labels


if __name__ == '__main__':

    opt_trajs = pd.read_csv("../b1_Main/SI_Trajectories/SI_ALL_Trajectories.csv")
    nonop_trajs = pd.read_csv("../c2_BC/BC_trajectories/BC_NonOPT_Trajectories.csv")
    dict_filename = '../Data/processed_data_dict_subset1_updated_negselection_normalized.pkl'

    dict = utils.load_dict(dict_filename)
    UPDATED_STATE_COLs = dict['UPDATED_STATE_COLs']
    ACTION_COLs = dict['ACTION_COLs']
    UPDATED_NEXT_STATE_COLs = dict['UPDATED_NEXT_STATE_COLs']
    ORI_DATA_COLs_WO_TRAINTEST = dict['ORI_DATA_COLs_WO_TRAINTEST']
    non_state_Act_cols = list(set(opt_trajs.columns) - (set(UPDATED_STATE_COLs).union(set(ACTION_COLs))))
    all_columns = copy.deepcopy(UPDATED_STATE_COLs)
    all_columns.extend(ACTION_COLs)
    all_columns.extend(non_state_Act_cols)

    states = opt_trajs[UPDATED_STATE_COLs].to_numpy()
    actions_optimal = opt_trajs[ACTION_COLs].to_numpy()
    actions_nonoptimal = nonop_trajs[ACTION_COLs].to_numpy()

    random_ordered_indices = list(np.random.permutation(len(states)))
    indices_opt = random_ordered_indices[:int(len(random_ordered_indices)/2)]
    indices_nonop = list(set(random_ordered_indices) - set(indices_opt))

    states_opt = states[indices_opt]
    states_nonop = states[indices_nonop]

    actions_opt = actions_optimal[indices_opt]
    actions_nonop = actions_nonoptimal[indices_nonop]

    states_combined = np.concatenate([states_opt, states_nonop])
    actions_combined = np.concatenate([actions_opt, actions_nonop])

    X_train, X_test, y_train, y_test = train_test_split(states_combined, actions_combined, test_size=0.2, random_state=42)

    input_size_ = X_train.shape[1]
    outut_size_ = y_train.shape[1]

    oversampled_train_data, oversampled_train_labels, oversampled_action_labels = handle_data_imbalance(X_train, y_train)

    # Build and train the policy network
    policy_net = build_policy_net(input_size_, outut_size_)
    policy_net, history = train_policy_net(policy_net, oversampled_train_data, oversampled_train_labels, X_test, y_test, num_epochs=10000)

    softmax_predictions = policy_net.predict(states)
    one_hot_predictions = np.zeros_like(softmax_predictions)
    one_hot_predictions[np.arange(len(softmax_predictions)), softmax_predictions.argmax(1)] = 1

    SemiOPT_Trjectories = copy.deepcopy(states)
    SemiOPT_Trjectories = pd.DataFrame(np.concatenate([SemiOPT_Trjectories, one_hot_predictions, opt_trajs[non_state_Act_cols]], axis=1), columns=all_columns)
    SemiOPT_Trjectories.to_csv('../c2_BC/BC_trajectories/SemiOPT_Trajectories.csv', index=False)

    print(' ------------- DONE ------------- ')
