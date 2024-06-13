import numpy as np
import matplotlib.pyplot as plt

class DecisionTree:
    def __init__(self) -> None:
        pass

    def compute_entropy(self, y):
        """
        Computes the entropy for the given labels.
        
        Args:
            y (ndarray): Numpy array indicating whether each example at a node is
                edible (`1`) or poisonous (`0`)
        
        Returns:
            entropy (float): Entropy at that node
        """
        if len(y) == 0:
            return 0.0
        
        p1 = np.mean(y == 1)
        if p1 == 0 or p1 == 1:
            return 0.0
        
        entropy = -p1 * np.log2(p1) - (1 - p1) * np.log2(1 - p1)
        return entropy

    def split_dataset(self, X, node_indices, feature):
        """
        Splits the data at the given node into left and right branches based on the feature.
        
        Args:
            X (ndarray): Data matrix of shape (n_samples, n_features)
            node_indices (list): List containing the active indices. I.e., the samples being considered at this step.
            feature (int): Index of the feature to split on
        
        Returns:
            left_indices (list): Indices with feature value == 1
            right_indices (list): Indices with feature value == 0
        """
        left_indices = [i for i in node_indices if X[i][feature] == 1]
        right_indices = [i for i in node_indices if X[i][feature] == 0]
        return left_indices, right_indices

    def compute_information_gain(self, X, y, node_indices, feature):
        """
        Compute the information gain of splitting the node on a given feature.
        
        Args:
            X (ndarray): Data matrix of shape (n_samples, n_features)
            y (array-like): List or ndarray with n_samples containing the target variable
            node_indices (ndarray): List containing the active indices. I.e., the samples being considered in this step.
            feature (int): Index of the feature to split on
        
        Returns:
            information_gain (float): Information gain computed
        """
        left_indices, right_indices = self.split_dataset(X, node_indices, feature)
        
        y_node = y[node_indices]
        y_left = y[left_indices]
        y_right = y[right_indices]
        
        node_entropy = self.compute_entropy(y_node)
        left_entropy = self.compute_entropy(y_left)
        right_entropy = self.compute_entropy(y_right)
        
        w_left = len(y_left) / len(y_node)
        w_right = len(y_right) / len(y_node)
        
        weighted_entropy = w_left * left_entropy + w_right * right_entropy
        information_gain = node_entropy - weighted_entropy
        
        return information_gain

    def get_best_split(self, X, y, node_indices):
        """
        Returns the optimal feature to split the node data.
        
        Args:
            X (ndarray): Data matrix of shape (n_samples, n_features)
            y (array-like): List or ndarray with n_samples containing the target variable
            node_indices (ndarray): List containing the active indices. I.e., the samples being considered in this step.
        
        Returns:
            best_feature (int): The index of the best feature to split
        """
        num_features = X.shape[1]
        best_feature = -1
        max_info_gain = 0
        
        for feature in range(num_features):
            info_gain = self.compute_information_gain(X, y, node_indices, feature)
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_feature = feature
            
        return best_feature

    def build_tree_recursive(self, X, y, node_indices, branch_name, max_depth, current_depth):
        """
        Build a tree using the recursive algorithm that splits the dataset into 2 subgroups at each node.
        This function just prints the tree.
        
        Args:
            X (ndarray): Data matrix of shape (n_samples, n_features)
            y (array-like): List or ndarray with n_samples containing the target variable
            node_indices (ndarray): List containing the active indices. I.e., the samples being considered in this step.
            branch_name (str): Name of the branch. ['Root', 'Left', 'Right']
            max_depth (int): Max depth of the resulting tree.
            current_depth (int): Current depth. Parameter used during recursive call.
        """
        if current_depth == max_depth or len(node_indices) == 0:
            print(f"{' ' * current_depth}{'-' * current_depth} {branch_name} leaf node with indices: {node_indices}")
            return

        best_feature = self.get_best_split(X, y, node_indices)
        if best_feature == -1:
            print(f"{' ' * current_depth}{'-' * current_depth} {branch_name} leaf node with indices: {node_indices}")
            return

        print(f"{'-' * current_depth} Depth {current_depth}, {branch_name}: Split on feature {best_feature}")
        
        left_indices, right_indices = self.split_dataset(X, node_indices, best_feature)
        
        self.build_tree_recursive(X, y, left_indices, "Left", max_depth, current_depth + 1)
        self.build_tree_recursive(X, y, right_indices, "Right", max_depth, current_depth + 1)
