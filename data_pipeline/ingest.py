import pandas as pd

def ingest_data(csv_path):
    """
    Ingest raw sensor data from a CSV file.
    Performs basic cleaning such as forward-filling missing values.
    """
    df = pd.read_csv(csv_path)
    df.fillna(method="ffill", inplace=True)
    return df

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python ingest.py path/to/sensor_data.csv")
    else:
        df = ingest_data(sys.argv[1])
        print("Data preview:")
        print(df.head())
