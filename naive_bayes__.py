
#-----------------------------------------------------------
# AUTHOR: Aaron Manoj
# FILENAME: naive_bayes.py
# SPECIFICATION: Categorical Naive Bayes
# FOR: CS 4210 - Assignment #2
# TIME SPENT: 50 minutes
#-----------------------------------------------------------

import pandas as pd
from collections import Counter, defaultdict

TRAIN_FILE = "weather_training.csv"
TEST_FILE  = "weather_test.csv"

def train_counts(df, feature_cols, target_col):
    class_counts = Counter(df[target_col])
    cond_counts = {feat: defaultdict(Counter) for feat in feature_cols}
    for _, row in df.iterrows():
        c = row[target_col]
        for feat in feature_cols:
            cond_counts[feat][c][row[feat]] += 1
    return class_counts, cond_counts

def posterior_no_smoothing(example, feature_cols, class_counts, cond_counts):
    total = sum(class_counts.values())
    numerators = {}
    for c, c_count in class_counts.items():
        p = c_count / total  # prior
        for feat in feature_cols:
            v = example[feat]
            n = cond_counts[feat][c][v]
            if n == 0:
                p = 0.0
                break
            p *= n / c_count
        numerators[c] = p
    z = sum(numerators.values())
    if z == 0:
        return {c: 0.0 for c in numerators}
    return {c: v / z for c, v in numerators.items()}

def main():
    train = pd.read_csv(TRAIN_FILE)
    test  = pd.read_csv(TEST_FILE)

    cols = list(train.columns)
    target_col   = cols[-1]
    feature_cols = cols[:-1]

    class_counts, cond_counts = train_counts(train, feature_cols, target_col)

    print("Day        Outlook     Temperature   Humidity   Wind    PlayTennis   Confidence")
    for _, row in test.iterrows():
        post = posterior_no_smoothing(row, feature_cols, class_counts, cond_counts)
        # choose class with highest posterior
        pred_label, conf = max(post.items(), key=lambda kv: kv[1])
        if conf >= 0.75:  # print only when confidence threshold satisfied
            print(f"{row['Day']:8}  {row['Outlook']:10}  {row['Temperature']:11}  "
                  f"{row['Humidity']:7}  {row['Wind']:6}  {pred_label:10}  {conf:.2f}")

if __name__ == "__main__":
    main()
