import numpy as np
import pandas as pd

def read_world(filepath):
    """
        Size: (Number of landmarks) x 3
        First column is index of landmark, second column is x coordinate, 
        third column is y coordinate
    """
    arr = np.loadtxt(filepath)
    return arr

def read_sensor_data(filename):
    # reads the sensor data as a dataframe
    df = pd.read_csv(filename, sep=" ", header=None, names=["0","1","2","3"],float_precision='round_trip')

    # creates a new column to store number of observations in the first sensor reading
    # Number of observations = second ODOMETRY entry - first ODOMETRY entry - 1
    df["4"] = 0
    odometry = 0
    # checking = 0
    for i in range(1,df.shape[0]):
        if df.iloc[i,0] == "ODOMETRY":
            # checking += 1
            df.iloc[odometry,4] = i - odometry - 1
            df.iloc[odometry+1,4] = i - odometry - 1
            odometry = i
    
    # Last one will not get updated
    # print(f"odometry count:{checking}")
    df.iloc[odometry,4] = (df.shape[0]) - odometry - 1
    df.iloc[odometry+1,4] = (df.shape[0]) - odometry - 1
    return df


if __name__ == "__main__":
    data_df = read_sensor_data("../../data/sensor_data.dat")
    print(data_df.head())
    abc = data_df.to_numpy()
    print(abc[:3,:])