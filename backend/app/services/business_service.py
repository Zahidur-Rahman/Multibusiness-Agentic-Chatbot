"""
Business and User management API for dynamic add/remove operations.
"""

from fastapi import APIRouter, HTTPException, status, Depends
from typing import List
from backend.app.models.business import BusinessConfig, BusinessConfigCreate, BusinessConfigUpdate
from backend.app.models.user import User, UserCreate, UserUpdate
from backend.app.services.mongodb_service import mongodb_service
from backend.app.utils import hash_password
import logging
from backend.app.auth.routes import require_role

router = APIRouter(prefix="/admin", tags=["business-admin"])
logger = logging.getLogger(__name__)

# =========================
# BUSINESS MANAGEMENT
# =========================

@router.post("/businesses", response_model=BusinessConfig)
async def add_business(business: BusinessConfigCreate, current_user=Depends(require_role('admin'))):
    """Dynamically add a new business."""
    try:
        created = await mongodb_service.create_business_config(business)
        return created
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error adding business: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/businesses/{business_id}")
async def remove_business(business_id: str, current_user=Depends(require_role('admin'))):
    """Dynamically remove a business."""
    try:
        deleted = await mongodb_service.delete_business_config(business_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Business not found")
        return {"message": f"Business '{business_id}' removed successfully"}
    except Exception as e:
        logger.error(f"Error removing business: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/businesses", response_model=List[BusinessConfig])
async def list_businesses(current_user=Depends(require_role('admin'))):
    """List all businesses."""
    return await mongodb_service.get_all_business_configs()

@router.get("/businesses/{business_id}", response_model=BusinessConfig)
async def get_business(business_id: str, current_user=Depends(require_role('admin'))):
    """Get a business by ID."""
    business = await mongodb_service.get_business_config(business_id)
    if not business:
        raise HTTPException(status_code=404, detail="Business not found")
    return business

# =========================
# USER MANAGEMENT
# =========================

@router.post("/users", response_model=User)
async def add_user(user: UserCreate, current_user=Depends(require_role('admin'))):
    """Dynamically add a new user."""
    hashed_password = hash_password(user.password)
    try:
        created = await mongodb_service.create_user(user, hashed_password)
        return created
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error adding user: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/users/{username}")
async def remove_user(username: str, current_user=Depends(require_role('admin'))):
    """Dynamically remove a user by username."""
    try:
        user = await mongodb_service.get_user_by_username(username)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        result = await mongodb_service._collections['users'].delete_one({"username": username})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="User not found")
        return {"message": f"User '{username}' removed successfully"}
    except Exception as e:
        logger.error(f"Error removing user: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/users", response_model=List[User])
async def list_users(current_user=Depends(require_role('admin'))):
    """List all users."""
    users = []
    cursor = mongodb_service._collections['users'].find({})
    async for doc in cursor:
        users.append(User(**doc))
    return users

@router.get("/users/{username}", response_model=User)
async def get_user(username: str, current_user=Depends(require_role('admin'))):
    """Get a user by username."""
    user = await mongodb_service.get_user_by_username(username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user 