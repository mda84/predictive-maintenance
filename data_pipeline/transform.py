import pandas as pd
from sklearn.preprocessing import StandardScaler

def transform_data(df, feature_columns):
    """
    Normalize the specified feature columns in the dataframe using StandardScaler.
    Returns the transformed dataframe and the fitted scaler.
    """
    scaler = StandardScaler()
    df_transformed = df.copy()
    df_transformed[feature_columns] = scaler.fit_transform(df_transformed[feature_columns])
    return df_transformed, scaler

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python transform.py path/to/sensor_data.csv col1 col2 ...")
    else:
        csv_path = sys.argv[1]
        features = sys.argv[2:]
        df = pd.read_csv(csv_path)
        df_transformed, scaler = transform_data(df, features)
        print("Transformed Data Preview:")
        print(df_transformed.head())
