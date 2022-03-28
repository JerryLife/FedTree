from sklearn.datasets import load_svmlight_file, dump_svmlight_file, make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

from train_test_split import save_data


def generate_overfit_dataset(save_path, seed=0):
    X, y = make_classification(n_samples=2000, n_features=50, n_redundant=40, n_informative=10, class_sep=0.2)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=seed)
    save_data(train_X, train_y, f"{save_path}.train")
    save_data(test_X, test_y, f"{save_path}.test")
    return train_X, train_y, test_X, test_y


def evaluate_overfit_dataset(train_X, train_y, test_X, test_y):
    classifier = GradientBoostingClassifier(learning_rate=1, n_estimators=30, max_depth=4)
    print("Training")
    classifier.fit(train_X, train_y)
    print("Testing")
    train_score = classifier.score(train_X, train_y)
    test_score = classifier.score(test_X, test_y)
    print(f"Training accuracy: {train_score}, test accuracy {test_score}")


if __name__ == '__main__':
    train_X, train_y, test_X, test_y = generate_overfit_dataset("../data/overfit")
    evaluate_overfit_dataset(train_X, train_y, test_X, test_y)


