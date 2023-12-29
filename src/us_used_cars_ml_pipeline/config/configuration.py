from us_used_cars_ml_pipeline.constants import *
from us_used_cars_ml_pipeline.utils.common import read_yaml
from us_used_cars_ml_pipeline import logger
from us_used_cars_ml_pipeline.entity.config_entity import (DataConversionConfig, 
                                                           CleanDataConfig,
                                                           DataPreparationConfig,
                                                           FeatureSelectionConfig,
                                                           ModelsTuningConfig,
                                                           StackingRegressorModelingConfig,
                                                           MetricsCalculationConfig)

class ConfigurationManager:
    """
    The ConfigurationManager class is responsible for reading and providing 
    configuration settings needed for various stages of the data pipeline.

    Attributes:
    - config (dict): Dictionary holding configuration settings from the config file.
    - params (dict): Dictionary holding parameter values from the params file.
    - schema (dict): Dictionary holding schema information from the schema file.
    """
    
    def __init__(self, 
                 config_filepath=CONFIG_FILE_PATH, 
                 params_filepath=PARAMS_FILE_PATH, 
                 schema_filepath=SCHEMA_FILE_PATH):
        """
        Initializes the ConfigurationManager with configurations, parameters, and schema.

        Parameters:
        - config_filepath (str): Filepath to the configuration file.
        - params_filepath (str): Filepath to the parameters file.
        - schema_filepath (str): Filepath to the schema file.
        """
        self.config = self._read_config_file(config_filepath, "config")
        self.params = self._read_config_file(params_filepath, "params")
        self.schema = self._read_config_file(schema_filepath, "schema")

    def _read_config_file(self, filepath: str, config_name: str) -> dict:
        """
        Reads and returns the content of a configuration file.

        Parameters:
        - filepath (str): The file path to the configuration file.
        - config_name (str): Name of the configuration (used for logging purposes).

        Returns:
        - dict: Dictionary containing the configuration settings.

        Raises:
        - Exception: An error occurred reading the configuration file.
        """
        try:
            return read_yaml(filepath)
        except Exception as e:
            logger.error(f"Error reading {config_name} file: {filepath}. Error: {e}")
            raise
            

    def get_data_conversion_config(self) -> DataConversionConfig:
        """
        Extracts and returns data conversion configuration settings as a DataConversionConfig object.

        Returns:
        - DataConversionConfig: Object containing data conversion configuration settings.

        Raises:
        - AttributeError: The 'data_conversion' attribute does not exist in the config file.
        """
        try:
            config = self.config.data_conversion
            return DataConversionConfig(
                path_to_csv_data=config.path_to_csv_data,
                path_to_parquet_data=config.path_to_parquet_data
            )
        except AttributeError as e:
            logger.error("The 'data_conversion' attribute does not exist in the config file.")
            raise e

    
    def get_clean_data_config(self) -> CleanDataConfig:
        """
        Extracts and returns data cleaning configuration settings as a CleanDataConfig object.

        Returns:
        - CleanDataConfig: Object containing data cleaning configuration settings.

        Raises:
        - AttributeError: The 'clean_data' attribute does not exist in the config file.
        """
        try:
            """
            config = self.config.clean_data
            return CleanDataConfig(
                path_to_cleaned_data=config.path_to_cleaned_data,
                path_to_raw_data=config.path_to_raw_data,
                features_to_encode=config.features_to_encode,
                features_with_nans=config.features_with_nans,
                features_to_drop=config.features_to_drop,
                glove_model=config.glove_model,
                data_types=config.data_types,
                popular_options=config.popular_options,
                label_encodings=config.label_encodings,
                kfold_encodings=config.kfold_encodings,
                rare_classes=config.rare_classes,
                seed=config.seed,
                n_folds=config.n_folds,
                rare_classes_count=config.rare_classes_count
            )
            """
            config = self.config['clean_data']
            return CleanDataConfig(
                path_to_cleaned_data=config['path_to_cleaned_data'],
                path_to_raw_data=config['path_to_raw_data'],
                features_to_encode=config['features_to_encode'],
                features_with_nans=config['features_with_nans'],
                features_to_drop=config['features_to_drop'],
                glove_model=config['glove_model'],
                data_types=config['data_types'],
                popular_options=config['popular_options'],
                label_encodings=config['label_encodings'],
                kfold_encodings=config['kfold_encodings'],
                rare_classes=config['rare_classes'],
                seed=config['seed'],
                n_folds=config['n_folds'],
                rare_classes_count=config['rare_classes_count']
            )
        except AttributeError as e:
            logger.error("The 'clean_data' attribute does not exist in the config file.")
            raise e


    def get_data_preparation_config(self) -> DataPreparationConfig:
        """
        Extracts and returns data preparation configuration settings as a DataPreparationConfig object.

        Returns:
        - DataPreparationConfig: Object containing data preparation configuration settings.

        Raises:
        - AttributeError: The 'data_preparation' attribute does not exist in the config file.
        """
        try:
            config = self.config.data_preparation
            return DataPreparationConfig(
                path_to_cleaned_data=config.path_to_cleaned_data,
                path_to_prepared_data=config.path_to_prepared_data,
                scaler_name=config.scaler_name,
                path_to_scalers=config.path_to_scalers,
                path_to_features_list=config.path_to_features_list,
                path_to_imputers=config.path_to_imputers,
                path_to_scores_file=config.path_to_scores_file,
                models=config.models
            )
        except AttributeError as e:
            logger.error("The 'data_preparation' attribute does not exist in the config file.")
            raise e


    def get_feature_selection_config(self) -> FeatureSelectionConfig:
        """
        Extracts and returns feature selection configuration settings as a FeatureSelectionConfig object.

        Returns:
        - FeatureSelectionConfig: Object containing feature selection configuration settings.

        Raises:
        - AttributeError: The 'feature_selection' attribute does not exist in the config file.
        """
        try:
            config = self.config.feature_selection
            return FeatureSelectionConfig(
                path_to_prepared_data=config.path_to_prepared_data,
                path_to_save_importances=config.path_to_save_importances,
                path_to_feature_selection_scores=config.path_to_feature_selection_scores,
                seed=config.seed,
                test_ratio=config.test_ratio,
                n_permutations=config.n_permutations,
                n_feats=config.n_feats,
                searching_params=config.searching_params,
                verbose=config.verbose,
                metrics=config.metrics,
                models=config.models
            )
        except AttributeError as e:
            logger.error("The 'feature_selection' attribute does not exist in the config file.")
            raise e

    

    def get_models_tuning_config(self) -> ModelsTuningConfig:
        """
        Extracts and returns configuration settings as a ModelsTuningConfig object.

        Returns:
        - ModelsTuningConfig: Object containing configuration settings for models tuning.

        Raises:
        - AttributeError: The 'models_tuning' attribute does not exist in the config file.
        """
        try:
            config = self.config.models_tuning
            return ModelsTuningConfig(
                path_to_prepared_data=config.path_to_prepared_data,
                path_to_importances=config.path_to_importances,
                path_to_feature_selection_scores=config.path_to_feature_selection_scores,
                path_to_parameters_grid=config.path_to_parameters_grid,
                path_to_best_models=config.path_to_best_models,
                path_to_best_params=config.path_to_best_params,
                path_to_scores=config.path_to_scores,
                metric=config.metric,
                tuner=config.tuner,
                seed=config.seed,
                test_ratio=config.test_ratio,
                n_folds=config.n_folds,
                models=config.models,
                n_feats=config.n_feats
            )
        except AttributeError as e:
            logger.error("The 'models_tuning' attribute does not exist in the config file.")
            raise e


    def get_stacking_regressor_modeling_config(self) -> StackingRegressorModelingConfig:
        """
        Extracts and returns configuration settings as a StackingRegressorModelingConfig object.

        Returns:
        - StackingRegressorModelingConfig: Object containing configuration settings for models tuning.

        Raises:
        - AttributeError: The 'stacking_regressor_modeling' attribute does not exist in the config file.
        """
        try:
            config = self.config.stacking_regressor_modeling
            return StackingRegressorModelingConfig(
                path_to_prepared_data=config.path_to_prepared_data,
                path_to_importances=config.path_to_importances,
                path_to_parameters_grid=config.path_to_parameters_grid,
                path_to_best_params=config.path_to_best_params,
                path_to_predictions=config.path_to_predictions,
                path_to_stacking_models=config.path_to_stacking_models,
                n_folds=config.n_folds,
                seed=config.seed,
                base_models_names=config.base_models_names,
                n_feats=config.n_feats,
                meta_model_name=config.meta_model_name,
                meta_model_params=config.meta_model_params,
                tuner=config.tuner,
                tuner_config=config.tuner_config,
                use_best_params=config.use_best_params
            )
        except AttributeError as e:
            logger.error("The 'stacking_regressor_modeling' attribute does not exist in the config file.")
            raise e
            
            
    def get_metrics_calculation_config(self) -> MetricsCalculationConfig:
        """
        Extracts and returns configuration settings as a MetricsCalculationConfig object.

        Returns:
        - MetricsCalculationConfig: Object containing configuration settings for metrics calculation.

        Raises:
        - AttributeError: The 'metrics_calculation' attribute does not exist in the config file.
        """
        try:
            config = self.config.metrics_calculation
            return MetricsCalculationConfig(
                path_to_raw_data=config.path_to_raw_data,
                raw_data_filename=config.raw_data_filename,
                path_to_predictions=config.path_to_predictions,
                predictions_filename=config.predictions_filename,
                path_to_metrics=config.path_to_metrics,
                metrics_filename=config.metrics_filename,
                metrics=config.metrics
            )
        except AttributeError as e:
            logger.error("The 'metrics_calculation' attribute does not exist in the config file.")
            raise e