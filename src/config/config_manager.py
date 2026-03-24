"""
Configuration Management Module
Handles loading and validation of YAML configuration files
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

class ConfigManager:
    """
    Manages configuration loading, validation, and access
    Provides a centralized way to handle all project settings
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
        self._validate_config()
        self._setup_logging()
    
    def _get_default_config_path(self) -> str:
        """Get path to default configuration file"""
        current_dir = Path(__file__).parent
        return str(current_dir.parent.parent / "configs" / "default_config.yaml")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            logging.info(f"Configuration loaded from: {self.config_path}")
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
    
    def _validate_config(self) -> None:
        """Validate configuration structure and values"""
        required_sections = ['data', 'cross_validation', 'model', 'preprocessing', 'evaluation']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate data configuration
        data_config = self.config['data']
        if 'path' not in data_config:
            raise ValueError("Data path not specified in configuration")
        
        # Validate model configuration
        model_config = self.config['model']
        if 'algorithm' not in model_config:
            raise ValueError("Model algorithm not specified")
        
        # Validate cross-validation configuration
        cv_config = self.config['cross_validation']
        if 'n_splits' not in cv_config:
            raise ValueError("Number of CV splits not specified")
        
        logging.info("Configuration validation passed")
    
    def _setup_logging(self) -> None:
        """Setup logging based on configuration"""
        log_config = self.config.get('logging', {})

        # Create logs directory if it doesn't exist
        log_file = log_config.get('file', 'parkinsons_analysis.log')
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        # Only configure root logger if it hasn't been set up already
        root_logger = logging.getLogger()
        if not root_logger.handlers:
            logging.basicConfig(
                level=getattr(logging, log_config.get('level', 'INFO')),
                format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
                handlers=[
                    logging.FileHandler(log_file),
                    logging.StreamHandler() if log_config.get('console', True) else logging.NullHandler()
                ]
            )
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key: Configuration key (e.g., 'model.parameters.n_estimators')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration section"""
        return self.config['data']
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration section"""
        return self.config['model']
    
    def get_cv_config(self) -> Dict[str, Any]:
        """Get cross-validation configuration section"""
        return self.config['cross_validation']
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Get preprocessing configuration section"""
        return self.config['preprocessing']
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration section"""
        return self.config['evaluation']
    
    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration section"""
        return self.config.get('output', {})
    
    def update(self, key: str, value: Any) -> None:
        """
        Update configuration value using dot notation
        
        Args:
            key: Configuration key (e.g., 'model.parameters.n_estimators')
            value: New value
        """
        keys = key.split('.')
        config = self.config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
        logging.info(f"Configuration updated: {key} = {value}")
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save current configuration to file
        
        Args:
            path: Path to save configuration. If None, uses original path.
        """
        save_path = path or self.config_path
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False, indent=2)
        
        logging.info(f"Configuration saved to: {save_path}")
    
    def create_output_directories(self) -> None:
        """Create output directories based on configuration"""
        output_config = self.get_output_config()
        
        directories = [
            output_config.get('results_dir', 'results'),
            output_config.get('models_dir', 'models'),
            output_config.get('plots_dir', 'plots'),
            output_config.get('reports_dir', 'reports')
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        logging.info("Output directories created")
    
    def __str__(self) -> str:
        """String representation of configuration"""
        return f"ConfigManager(config_path='{self.config_path}')"
    
    def __repr__(self) -> str:
        """Detailed representation of configuration"""
        return f"ConfigManager(config_path='{self.config_path}', sections={list(self.config.keys())})"
