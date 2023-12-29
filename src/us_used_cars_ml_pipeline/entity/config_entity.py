from dataclasses import dataclass



@dataclass(frozen=True)
class DataConversionConfig:
    """
    Configuration for the data ingestion process.
    
    Attributes:
    - path_to_csv_data: Path to raw data as csv.
    - path_to_parquet_data: Path to raw data as parquet.
    """
    
    path_to_csv_data: str
    path_to_parquet_data: str  

        

@dataclass(frozen=True)
class CleanDataConfig:
    """
    Configuration for the data cleaning process.
    
    Attributes:
    - root_dir: The directory where data ingestion artifacts should be stored.
    - source_URL: The source URL from which the dataset is to be downloaded.
    - clean_data_URL: The path in HDFS where the clean data file should be stored.
    """
    
    path_to_cleaned_data: str  # Directory for data ingestion related artifacts
    path_to_raw_data: str  # URL for the source dataset
    features_to_encode: str
    features_with_nans: str
    features_to_drop: str
    glove_model: str
    data_types: str
    popular_options: str
    label_encodings: str
    kfold_encodings: str
    rare_classes: str
    seed: int
    n_folds: int
    rare_classes_count: list
    


@dataclass(frozen=True)
class DataPreparationConfig:
    """
    Configuration for the data preparation process.
    
    Attributes:
    - path_to_cleaned_data: The path in HDFS where cleaned data is stored.
    - path_to_prepared_data: 
    - scaler_name: Name of a scaler from Spark ML, that should be applied
    - path_to_scalers: The path to scalers (for different sets of features) where they should be stored.
    - path_to_features_list: The path to the fille with the list of features to be considered.
    - path_to_imputers: The path to directory where imputer models must be stored.
    - path_to_scores_file: The path to yaml file where to store scores
    - models: Dictionary with Imputer Models names and their parameters 
    """

    path_to_cleaned_data: str
    path_to_prepared_data: str
    scaler_name: str
    path_to_scalers: str
    path_to_features_list: str
    path_to_imputers: str
    path_to_scores_file: str
    models: dict



@dataclass(frozen=True)
class FeatureSelectionConfig:
    """
    Configuration for the feature selection process.
    
    Attributes:
    - path_to_prepared_data: 
    - path_to_save_importances: 
    - path_to_feature_selection_scores:
    - seed: Seed for shuffling process
    - test_ratio: Test data fraction
    - n_permutations: Number of permutations used in PermutationImportance selector
    - verbose: Level of extra logs (0 - no extra log, 
                                    1 - showing importance values for each feature,
                                    2 - showing importance values for each feature & scores for each feature and permutation)
    - n_feats: Number of features in data
    - searching_params: Dictionary with parameters for searching stage for specified ML models.
    - metrics: List of metrics to be calculated, when finding the best top features set based on importances
    - models: ML models to be considerd.
    """

    path_to_prepared_data: str
    path_to_save_importances: str
    path_to_feature_selection_scores: str
    seed: int
    test_ratio: float
    n_permutations: int
    n_feats: int
    searching_params: dict
    verbose: int
    metrics: list
    models: dict



@dataclass(frozen=True)
class ModelsTuningConfig:
    """
    Configuration for the models tuning process.
    
    Attributes:
    - path_to_prepared_data: Path to prepared data in HDFS
    - path_to_importances: Path to yaml files with calculated features importances for different models
    - path_to_feature_selection_scores: Path to yaml files with scores for different sets of the most important features for different models
    - path_to_parameters_grid: Path to yaml files with parameters grid for different models
    - path_to_best_models: Path in HDFS where to save tuned models
    - path_to_best_params: Path where to save best set of parameters for each considered model
    - path_to_scores: Path where to save tuning scores for each considered model
    - metric: Name of a metric to be used
    - tuner: Tuner to be used (TVSTuner or CVTuner)
    - seed: Seed to fix random state of tuners.
    - test_ratio: Test data fraction parameter for TVSTuner
    - n_folds: Number of folds parameter for CVTuner
    - models: List of models to be tuned
    - n_feats: List with number of top features to be selected for each each model
    """

    path_to_prepared_data: str
    path_to_importances: str
    path_to_feature_selection_scores: str
    path_to_parameters_grid: str
    path_to_best_models: str
    path_to_best_params: str
    path_to_scores: str
    metric: str
    tuner: str
    seed: int
    test_ratio: float
    n_folds: int
    models: list
    n_feats: list



@dataclass(frozen=True)
class StackingRegressorModelingConfig:
    """
    Configuration for the models tuning process.
    
    Attributes:
    - path_to_prepared_data: Path to prepared data in HDFS
    - path_to_importances: Path to yaml files with calculated features importances for different models
    - path_to_parameters_grid: Path to yaml files with parameters grid for different models
    - path_to_best_params: Path where to save best set of parameters for each considered model
    - path_to_predictions: 
    - path_to_stacking_models: 
    - n_folds: Number of folds
    - seed: Seed to fix random state
    - base_models_names: List of models to be used as base models
    - n_feats: List with number of top features to be selected for each each base model
    - meta_model_name: Name of the meta model
    - meta_model_params: Parameters of the meta model
    - tuner: Name of a tuner to be used to tune meta model
    - tuner_config: Configuration parameters of a tuner
    """

    path_to_prepared_data: str
    path_to_importances: str
    path_to_parameters_grid: str
    path_to_best_params: str
    path_to_predictions: str
    path_to_stacking_models: str
    n_folds: int
    seed: int
    base_models_names: list
    n_feats: list
    meta_model_name: str
    meta_model_params: dict  
    tuner: str
    tuner_config: dict
    use_best_params: int
        

     
@dataclass(frozen=True)
class MetricsCalculationConfig:
    """
    Configuration for the metrics calculation process.
    
    Attributes:
    - path_to_raw_data: 
    - raw_data_filename: 
    - path_to_predictions: 
    - predictions_filename: 
    - path_to_metrics: 
    - metrics_filename: 
    - metrics: 
    """

    path_to_raw_data: str
    raw_data_filename: str
    path_to_predictions: str
    predictions_filename: str
    path_to_metrics: str
    metrics_filename: str
    metrics: list