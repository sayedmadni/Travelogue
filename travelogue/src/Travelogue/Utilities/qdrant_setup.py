"""
Qdrant setup utilities
"""
import os
import yaml
from qdrant_client import QdrantClient

# Default port
PORT = 6333

def load_config():
    """Load configuration from config.yaml"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up: Utilities -> Travelogue -> src -> travelogue
    travelogue_dir = os.path.join(current_dir, '..', '..', '..')
    config_path = os.path.join(os.path.abspath(travelogue_dir), 'config.yaml')
    
    if not os.path.exists(config_path):
        # Try alternative: look for config.yaml in parent directories
        search_dir = current_dir
        for _ in range(6):
            potential_path = os.path.join(search_dir, 'config.yaml')
            if os.path.exists(potential_path):
                config_path = potential_path
                break
            search_dir = os.path.dirname(search_dir)
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        # Return default config
        return {
            "vector-db": {"host": "localhost", "port": 6333}
        }

def start_vector_db():
    """
    Ensure Qdrant is accessible. 
    This is a placeholder - assumes Qdrant is already running.
    In production, you might want to start a Docker container here.
    """
    config = load_config()
    host = config.get("vector-db", {}).get("host", "localhost")
    port = config.get("vector-db", {}).get("port", 6333)
    
    # Try to connect to verify Qdrant is running
    try:
        client = QdrantClient(host=host, port=port)
        # Simple health check
        client.get_collections()
        return True
    except Exception as e:
        print(f"Warning: Could not connect to Qdrant at {host}:{port}")
        print(f"Make sure Qdrant is running. Error: {e}")
        return False

# Update PORT from config
config = load_config()
PORT = config.get("vector-db", {}).get("port", 6333)

