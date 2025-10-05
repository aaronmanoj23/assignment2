
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

TRAIN_FILES = [
    "contact_lens_training_1.csv",
    "contact_lens_training_2.csv",
    "contact_lens_training_3.csv",
]
TEST_FILE = "contact_lens_test.csv"

def load_csv(path):
    return pd.read_csv(path)

def prepare_encoders(train_df, test_df, feature_cols):
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(pd.concat([train_df[feature_cols], test_df[feature_cols]], axis=0))
    return enc

def encode_features(enc, df, feature_cols):
    return enc.transform(df[feature_cols])

def main():
    # Load test once
    test_df = load_csv(TEST_FILE)

    # assume last column is the target
    all_cols = list(test_df.columns)
    target_col = all_cols[-1]
    feature_cols = all_cols[:-1]

    # keep raw test labels
    y_test = test_df[target_col].values

    results = []
    for train_path in TRAIN_FILES:
        accs = []
        for run in range(10):
            train_df = load_csv(train_path)

            all_cols_train = list(train_df.columns)
            target_col_train = all_cols_train[-1]
            feature_cols_train = all_cols_train[:-1]
            if feature_cols_train != feature_cols:
                # Ensure same feature order
                train_df = train_df[feature_cols + [target_col_train]]

            enc = prepare_encoders(train_df, test_df, feature_cols)
            X_train = encode_features(enc, train_df, feature_cols)
            X_test = encode_features(enc, test_df, feature_cols)

            y_train = train_df[target_col_train].values

            clf = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=run)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accs.append(accuracy_score(y_test, y_pred))

        avg_acc = float(np.mean(accs))
        results.append((train_path, avg_acc))

    print("Average accuracies over 10 runs (max_depth=5, criterion=entropy):")
    for path, acc in results:
        print(f"{path}: {acc:.4f}")

if __name__ == "__main__":
    main()
