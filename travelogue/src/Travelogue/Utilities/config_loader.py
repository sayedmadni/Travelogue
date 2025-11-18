"""
Config loader utility - loads configuration from config.yaml
"""
import os
import yaml
import sys

def load_config():
    """
    Load configuration from config.yaml file
    """
    # Find config.yaml by looking in the travelogue directory
    # This file is at: travelogue/src/Travelogue/Utilities/config_loader.py
    # config.yaml is at: travelogue/config.yaml
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up: Utilities -> Travelogue -> src -> travelogue
    travelogue_dir = os.path.join(current_dir, '..', '..', '..')
    config_path = os.path.join(os.path.abspath(travelogue_dir), 'config.yaml')
    
    if not os.path.exists(config_path):
        # Try alternative: look for config.yaml in parent directories
        search_dir = current_dir
        for _ in range(6):  # Search up to 6 levels up
            potential_path = os.path.join(search_dir, 'config.yaml')
            if os.path.exists(potential_path):
                config_path = potential_path
                break
            search_dir = os.path.dirname(search_dir)
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    else:
        # Return minimal default config
        print(f"⚠️ Warning: Could not find config.yaml. Using default config.", file=sys.stderr)
        return {
            "clip": {"model": "openai/clip-vit-base-patch32", "collection_name": "tree_embeddings"},
            "siglip2": {"model": "google/siglip2-base-patch16-224", "collection_name": "tree_siglip2_embeddings"},
            "deepface": {"collection_name": "faces_collection", "threshold": 0.4},
            "vector-db": {"host": "localhost", "port": 6333},
            "datasets": {}
        }

# Create a module-level config that can be imported
config = load_config()

