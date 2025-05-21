from math import sqrt
from pathlib import Path
import pickle
from typing import Optional
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import csr_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

"""
Initial script to read data from parquet files and build a prediction model.  
"""
# Setting the default directory:
CURRENT_DIR = Path(__file__).parent

# Output logs for homework purposes:
log_path = CURRENT_DIR / "homework"
logger.add(log_path / "output.log", rotation="1 MB", level="INFO")


def read_parquet_file(filename: str) -> pd.DataFrame:
    """ Function to read parquet files from the data folder"""
    data_path = CURRENT_DIR / "data" / filename
    
    if not data_path.exists():
        raise FileNotFoundError(f"File not found: {data_path}")
    
    logger.info(f"Reading Parquet file: {data_path}")
    return pd.read_parquet(data_path, engine="pyarrow")


def percentage_of_outliers(
    df: pd.DataFrame,
    column: str = "duration",
    method: str = "iqr",
    threshold: float = 1.5
) -> float:
    """
    Calculates the percentage of outliers in the specified column.
    """
    if column not in df.columns:
        raise KeyError(f"{column} not found in DataFrame")

    series = df[column]

    if method == "iqr":
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
    elif method == "std":
        mean = series.mean()
        std = series.std()
        lower = mean - threshold * std
        upper = mean + threshold * std
    else:
        raise ValueError("Method must be 'iqr' or 'std'")

    mask = (series < lower) | (series > upper)
    outlier_percentage = mask.mean() * 100

    logger.info(f"Outlier percentage in '{column}' using {method}: {outlier_percentage:.2f}%")
    return outlier_percentage


def add_trip_duration_and_slice(df: pd.DataFrame, 
                                pickup_col: str, 
                                dropoff_col: str,
                                min_duration: Optional[float] = 1, 
                                max_duration: Optional[float] = 60) -> pd.DataFrame:
    """ 
    Function to add a 'duration' column (in minutes) to df based on dropoff - pickup,
    then filter rows where duration is between min_duration and max_duration.
    """
    # Check if columns exist
    if pickup_col not in df.columns or dropoff_col not in df.columns:
        raise KeyError(f"Columns {pickup_col} and/or {dropoff_col} not found in DataFrame")

    logger.info("Columns exsist in the dataframe, converting to datetime if not done ...")
    df[pickup_col] = pd.to_datetime(df[pickup_col], errors='coerce')
    df[dropoff_col] = pd.to_datetime(df[dropoff_col], errors='coerce')
    
    logger.info("Creating the trip duration column in minutes")
    df['duration'] = (df[dropoff_col] - df[pickup_col]).dt.total_seconds() / 60
    logger.info(f"The standard deviation of the duration column is before slice is: {df['duration'].std()}")
    
    # Assuming you already have df_with_duration
    percent_outliers = percentage_of_outliers(df, column="duration", method="iqr", threshold=1.5)
    logger.info(f"The percent of outliers derived from the column duration is: {percent_outliers}")

    # Filter if both min_duration and max_duration are not None
    if min_duration is not None and max_duration is not None:
        df = df[(df['duration'] >= min_duration) & (df['duration'] <= max_duration)]
        
    logger.info(f"The standard deviation of the duration column after slice is: {df['duration'].std()}")
    return df

def enforce_column_types(
    df: pd.DataFrame,
    categorical_cols: list[str],
    numerical_cols: list[str]
) -> pd.DataFrame:
    """
    Enforce column types on the DataFrame:
    - Converts categorical_cols to strings
    - Converts numerical_cols to numeric (float)
    """
    # Cast categorical columns to string
    for col in categorical_cols:
        if col in df.columns:
            df.loc[:, col] = df[col].astype(str)
        else:
            logger.error(f"Warning: Categorical column '{col}' not found in DataFrame")
    
    # Cast numerical columns to numeric dtype (float)
    for col in numerical_cols:
        if col in df.columns:
            df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')
        else:
            logger.error(f"Warning: Numerical column '{col}' not found in DataFrame")

    return df

def preprocess_features_and_target(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    categorical: list[str],
    numerical: list[str],
    target: str
) -> tuple[csr_matrix, csr_matrix, np.ndarray, np.ndarray, DictVectorizer]:
    
    logger.info("Starting preprocessing of train and validation data")

    if 'PU_DO' in categorical:
        logger.debug("Creating 'PU_DO' composite feature from PULocationID and DOLocationID")
        df_train = df_train.copy()
        df_val = df_val.copy()
        df_train['PU_DO'] = df_train['PULocationID'].astype(str) + '_' + df_train['DOLocationID'].astype(str)
        df_val['PU_DO'] = df_val['PULocationID'].astype(str) + '_' + df_val['DOLocationID'].astype(str)


    logger.info(f"Vectorizing features: categorical={categorical}, numerical={numerical}")
    dv = DictVectorizer()

    train_dicts = df_train[categorical + numerical].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)
    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"Number of features after one-hot encoding: {X_train.shape[1]}")


    val_dicts = df_val[categorical + numerical].to_dict(orient='records')
    X_val = dv.transform(val_dicts)
    logger.info(f"X_val shape: {X_val.shape}")

    y_train = df_train[target].values
    y_val = df_val[target].values
    logger.info(f"y_train shape: {y_train.shape}, y_val shape: {y_val.shape}")

    return X_train, X_val, y_train, y_val, dv


def train_and_serialize_linear_regression(
    X_train: csr_matrix,
    y_train: np.ndarray,
    X_val: csr_matrix,
    y_val: np.ndarray,
    dv: DictVectorizer,
    model_name: str = "lin_reg.bin"
) -> float:
    """
    Trains a linear regression model, evaluates it on the validation set, and serializes the model and vectorizer.
    """
    logger.info("Training Linear Regression model...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    logger.info("Predicting on training set...")
    y_pred_train = lr.predict(X_train)
    train_rmse = sqrt(mean_squared_error(y_train, y_pred_train))
    logger.info(f"Validation RMSE: {train_rmse:.4f}")

    logger.info("Predicting on validation set...")
    y_pred = lr.predict(X_val)
    rmse = sqrt(mean_squared_error(y_val, y_pred))
    logger.info(f"Validation RMSE: {rmse:.4f}")

    model_dir = CURRENT_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / model_name  

    logger.info(f"Serializing model to {model_path}")
    with open(model_path, 'wb') as f_out:
        pickle.dump((dv, lr), f_out)

    return rmse

# To use the script:
if __name__ == "__main__":
    # 1. Reading parquet file:
    df = read_parquet_file(filename="yellow_tripdata_2023-01.parquet")
    logger.info(f"The shape of the dataframe is: {df.shape}")
    logger.info(f"The initial dataframe is: {df.head()}")
    
    # 2. Adding duration column and slicing the dataframe based on the max or min duration. 
    df_with_duration = add_trip_duration_and_slice(
    df,
    pickup_col='tpep_pickup_datetime',
    dropoff_col='tpep_dropoff_datetime',
    min_duration=1,
    max_duration=60,
    )
    logger.info(f"The shape of the dataframe is: {df_with_duration.shape}")
    logger.info(f"The Columnd trip duration described is: {df_with_duration.duration.describe()}")

    # 3.  
    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']

    df_enforced = enforce_column_types(df_with_duration, categorical, numerical)
    
    # 4 Applying the same steps for the validation dataframe:
    logger.info("Reading validation data")
    df_val = read_parquet_file(filename="yellow_tripdata_2023-02.parquet")
    df_val_with_duration = add_trip_duration_and_slice(
        df_val,
        pickup_col='tpep_pickup_datetime',
        dropoff_col='tpep_dropoff_datetime',
        min_duration=1,
        max_duration=60,
    )
    df_val_enforced = enforce_column_types(df_val_with_duration, categorical, numerical)
    logger.info(f"Filtered validation dataframe shape: {df_val_with_duration.shape}")
    logger.debug(df_val_with_duration.duration.describe())
    
    # 5. 
    target = 'duration'

    X_train, X_val, y_train, y_val, dv = preprocess_features_and_target(
        df_train=df_enforced, 
        df_val=df_val_enforced, 
        categorical=categorical, 
        numerical=numerical, 
        target=target
    )
    logger.success("Feature preprocessing completed successfully")
    
    
    rmse = train_and_serialize_linear_regression(X_train, y_train, X_val, y_val, dv)
    logger.info(f"Final RMSE: {rmse:.4f}")