import numpy as np
import Utils.a3_utils_updated_division as utils
import tensorflow as tf
from tensorflow.keras import regularizers
from sklearn.utils import shuffle
import bc_Utils as BCUtils
import pandas as pd


def build_policy_net(input_size_, output_size):
    input_size = 256
    # Define a neural network for the policy
    policy_net = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_size_,)),
        tf.keras.layers.Dense(int(input_size/2), activation="tanh", kernel_regularizer=regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(int(input_size), activation="relu", kernel_regularizer=regularizers.l2(0.002)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(int(input_size / 2), activation="relu", kernel_regularizer=regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(int(output_size * 2), activation="relu", kernel_regularizer=regularizers.l2(0.002)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(output_size, activation="softmax")
    ])

    # Compile the policy network
    # SGD was better than Adam
    print('SGD')
    policy_net.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9), loss="categorical_crossentropy", metrics=['accuracy'])
    return policy_net


def train_policy_net(policy_net, train_data, train_labels, test_data, test_labels, num_epochs=10000, batch_size=1024):
    print(batch_size)
    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    for i in range(num_epochs):
        # Randomly sample a batch of data points and labels from the training set
        idxs = np.random.choice(len(train_data), size=batch_size, replace=True)
        x = train_data[idxs]
        y = train_labels[idxs]

        # Train the policy network on the batch of data points
        policy_net.fit(x, y, epochs=1, validation_split=0.2, callbacks=[early_stop_callback], verbose=0)

        # Print the loss and accuracy every 100 steps
        if i % 100 == 0:
            loss, train_accuracy = policy_net.evaluate(train_data, train_labels, verbose=0)
            test_loss, test_accuracy = policy_net.evaluate(test_data, test_labels, verbose=0)
            print(f"Step {i}\ttrain_loss:\t{loss:.4f}\ttrain_accuracy:\t{train_accuracy:.4f}\ttest_loss:\t{test_loss:.4f}\ttest_accuracy:\t{test_accuracy:.4f}")
    return policy_net


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


if __name__ == "__main__":

    data_file = '../Data/<data_file>.pkl'   # processed and normalized data file
    D = utils.load_dict(data_file)

    exp_data = D['exp_data']
    Train_optml_data = D['train_policy_df_optml']
    Test_optml_data = D['test_df_optml']

    Train_demo_observations = np.asarray(Train_optml_data[D['UPDATED_STATE_COLs']])
    Train_demo_actions = np.asarray(Train_optml_data[D['ACTION_COLs']])
    Test_demo_observations = np.asarray(Test_optml_data[D['UPDATED_STATE_COLs']])
    Test_demo_actions = np.asarray(Test_optml_data[D['ACTION_COLs']])

    obs_dim = len(Train_demo_observations[1])
    act_dim = len(Train_demo_actions[1])
    oversampled_train_data, oversampled_train_labels, oversampled_action_labels = handle_data_imbalance(Train_demo_observations, Train_demo_actions)

    # Build and train the policy network
    policy_net = build_policy_net(obs_dim, act_dim)
    policy_net = train_policy_net(policy_net, oversampled_train_data, oversampled_train_labels, Test_demo_observations, Test_demo_actions)

    # Test the policy network on the test set
    # UPDATED TO CRETE TRAJECTORIES
    data_nonop = D['train_policy_df_nonop']
    data_nonop['type'] = 'NOP'
    data_op = D['train_policy_df_optml']
    data_op['type'] = 'OPT'

    state_cols = D['UPDATED_STATE_COLs']
    action_cols = D['ACTION_COLs']
    merged_df = pd.concat([exp_data, data_nonop, data_op])
    # A simple check to see if steps are optimal and nonoptimal intermittently for the same patient. Which is a main hypothesis of oor approach
    result = BCUtils.check_for_intermittent_opt_nonopt_steps(merged_df)
    print(f'\nDoes training data have opt and nonopt steps intermittently for given patients? ', result)

    df_altered = BCUtils.create_trajectories('OPT', merged_df, policy_net, state_cols, action_cols)
    df_altered.to_csv('./BC_trajectories/BC_OPT_Trajectories.csv', index=False)

    print('................................... Done, trajectories are saved.  ...................................')
