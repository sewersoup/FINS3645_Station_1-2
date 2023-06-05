import pandas as pd
import matplotlib as plt
import numpy as np
nRowsRead = 100000

def Station1_loadData():
    """
    Process data load post extract and transfer phases
    :return: df cleansed dataset
    """
    ## LOAD VARIATION #1 pandas.read_csv: Read a comma-separated values (csv) file into DataFrame.
    #file_path = "/kaggle/input/smart-home-dataset-with-weather-information/HomeC.csv"
    #df = pd.read_csv(file_path, low_memory=False)
    #df.info()

    ## LOAD VARIATION #2 direct file form your internal file system
    df = pd.read_csv('/Users/devynmaseimilian/Desktop/FINS3645/HomeC.csv', delimiter=',', nrows=nRowsRead)
    nRow, nCol = df.shape
    print(f'There are {nRow} rows and {nCol} columns')
    print(df.head(5))
    print(df.tail(5))
    print(df.info())

    ## CLEANSING clean dataset to be ready for handover to Station #2
    ## your task in Station #1 is to provide authentic dataset for further processing
    # this line of code will delete very last row of the original data as there is "/" error
    #df = df[0:-1]
    #df.tail(5)

    #Rename columns by replacing [kW] with ''
    df.columns = [col.replace(' [kW]', '') for col in df.columns]
    print(df.columns)

    #Convert unix time column into readable format
    time_index = pd.date_range('2016-01-01 05:00', periods=len(df),  freq='min')
    time_index = pd.DatetimeIndex(time_index)
    df = df.set_index(time_index)
    df = df.drop(['time'], axis=1)

    #Control for any duplicates
    df['use'].resample('D').mean().plot()
    # plt.show()
    df['House overall'].resample('D').mean().plot()
    # plt.show()
    df = df.drop(columns=['House overall'])

    return df

def featuresEngineeringBase(df):
    """
    Receive cleaned data from Station #1 process all relevant features
    :param df: input clean data streams
    :return: df: return relevant features
    """
    # Create aggregate features either as a sum or as average, based on detailed data observations
    df['sum_Furnace'] = df[['Furnace 1', 'Furnace 2']].sum(axis=1)
    df['avg_Kitchen'] = df[['Kitchen 12', 'Kitchen 14', 'Kitchen 38']].mean(axis=1)
    df = df.drop(['Kitchen 12', 'Kitchen 14', 'Kitchen 38'], axis=1)
    dft = df.drop(['Furnace 1', 'Furnace 2'], axis=1)
    print(df.columns)

    # Visualization view features in visual forms
    # Visualize ALL core features and data attributes e.g. "use" and "temperature"
    df['use'].plot(label="Energy Kw")
    # plt.legend()
    # plt.show()

    df['temperature'].plot(label="Temp")
    # plt.legend()
    # plt.show()

    # Investigate each relevant feature for patterns and utility or use in the overall dataset
    df['Microwave'].resample("h").mean().iloc[:24].plot()
    # plt.show()
    # df.groupby(df.index.hour).mean()['Microwave'].plot(xticks=np.arange(24)).set(xlabel='Daily Hours',
    #                                                                              ylabel='Microwave Usage (kW)')
    # plt.show()

    return df


# def LSTM_RNN_featuresBase(df):
#     # Setup Neural Network Data Features
#     weather_features = df[
#         ['temperature', 'humidity', 'visibility', 'windSpeed', 'pressure', 'windBearing', 'precipIntensity',
#          'precipProbability']]
#     energy_use = df['use']
#
#     x_train = weather_features[:7000]
#     y_train = energy_use[:7000]
#
#     x_test = weather_features[7000:]
#     y_test = energy_use[7000:]
#
#     x_train = np.reshape(x_train.values, (x_train.shape[0], x_train.shape[1], 1))
#     x_test = np.reshape(x_test.values, (x_test.shape[0], x_test.shape[1], 1))
#
#     return x_train, x_test, y_train, y_test
#
#
# def LSTM_RNN_featuresAdvanced(df):
#     # Setup Neural Network Data Features
#     weather_features = df[
#         ['temperature', 'humidity', 'visibility', 'windSpeed', 'pressure', 'windBearing', 'precipIntensity',
#          'precipProbability']]
#     energy_use = df['use']
#     weather_features = weather_features['2016-01-02 05:00:00':'2016-12-02 05:00:00']
#     weather_features['yesterday_use'] = energy_use['2016-01-01 05:00:00':'2016-12-01 05:00:00'].values
#     energy_use = df['use']['2016-01-02 05:00:00':'2016-12-02 05:00:00']
#
#     x_train = weather_features[:5000]
#     y_train = energy_use[:5000]
#
#     x_test = weather_features[5000:]
#     y_test = energy_use[5000:]
#
#     x_train = np.reshape(x_train.values, (x_train.shape[0], x_train.shape[1], 1))
#     x_test = np.reshape(x_test.values, (x_test.shape[0], x_test.shape[1], 1))
#
#     return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    # STATION 1
    df = Station1_loadData()
    # STATION 2
    df1 = featuresEngineeringBase(df)
    print(df1)