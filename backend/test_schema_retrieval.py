#!/usr/bin/env python3
"""
Test script to verify schema retrieval functionality
"""

import asyncio
import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.services.vector_search import FaissVectorSearchService
from app.services.mongodb_service import get_mongodb_service

async def test_schema_retrieval():
    """Test schema retrieval for different queries"""
    
    print("Testing schema retrieval...")
    
    # Initialize services
    mongo_service = await get_mongodb_service()
    vector_search = FaissVectorSearchService()
    
    # Test business IDs
    business_ids = ["library", "resturent"]
    
    for business_id in business_ids:
        print(f"\n=== Testing business: {business_id} ===")
        
        # Get all schemas
        schemas = await mongo_service.get_business_schemas(business_id)
        print(f"Total schemas found: {len(schemas)}")
        
        for schema in schemas:
            print(f"  - {schema.table_name}: {schema.schema_description}")
        
        # Test vector search for different queries
        test_queries = ["customer", "menu", "book", "order", "data"]
        
        for query in test_queries:
            print(f"\n--- Testing query: '{query}' ---")
            results = await vector_search.search_schemas(business_id, query, top_k=3)
            print(f"Results found: {len(results)}")
            
            for result in results:
                print(f"  - {result.get('table_name', 'Unknown')}: {result.get('schema_description', 'No description')}")

if __name__ == "__main__":
    asyncio.run(test_schema_retrieval()) 