"""
Basic tests for the Multi-Business Conversational Chatbot.
"""

import pytest
import sys
import os

# Add the backend directory to the Python path
backend_path = os.path.join(os.path.dirname(__file__), '..', 'backend')
sys.path.insert(0, backend_path)

# Import after adding to path
from app.config import get_settings, Settings

def test_settings_loading():
    """Test that settings can be loaded correctly"""
    settings = get_settings()
    assert isinstance(settings, Settings)
    assert settings.server.host == "0.0.0.0"
    assert settings.server.port == 8000
    assert settings.database.host == "localhost"

def test_business_configs():
    """Test that business configurations are loaded"""
    settings = get_settings()
    assert "default" in settings.business_configs
    assert len(settings.business_ids) >= 1

def test_feature_flags():
    """Test that feature flags are properly configured"""
    settings = get_settings()
    assert hasattr(settings.features, 'enable_multi_business')
    assert hasattr(settings.features, 'enable_vector_search')
    assert hasattr(settings.features, 'enable_schema_discovery')

def test_database_url_generation():
    """Test database URL generation"""
    from app.config import get_database_url
    url = get_database_url("default")
    assert "postgresql://" in url
    assert "localhost" in url

if __name__ == "__main__":
    # Run basic tests
    test_settings_loading()
    test_business_configs()
    test_feature_flags()
    test_database_url_generation()
    print("✅ All basic tests passed!") 