from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List

from ..database import get_db
from ..models.training_example import TrainingExample
from ..schemas.training_example import (
    TrainingExampleCreate,
    TrainingExampleUpdate,
    TrainingExampleResponse,
    TrainingStats,
)

router = APIRouter(prefix="/api/ai-trainer", tags=["ai-trainer"])


@router.get("/stats", response_model=TrainingStats)
def get_training_stats(db: Session = Depends(get_db)):
    total = db.query(TrainingExample).count()

    category_counts = (
        db.query(TrainingExample.category, func.count(TrainingExample.id))
        .group_by(TrainingExample.category)
        .all()
    )

    by_category = {cat: count for cat, count in category_counts}

    return TrainingStats(total_examples=total, by_category=by_category)


@router.get("/examples", response_model=List[TrainingExampleResponse])
def get_examples(category: str = None, db: Session = Depends(get_db)):
    query = db.query(TrainingExample)
    if category:
        query = query.filter(TrainingExample.category == category)
    return query.order_by(TrainingExample.created_at.desc()).all()


@router.get("/examples/{example_id}", response_model=TrainingExampleResponse)
def get_example(example_id: int, db: Session = Depends(get_db)):
    example = db.query(TrainingExample).filter(TrainingExample.id == example_id).first()
    if not example:
        raise HTTPException(status_code=404, detail="Example not found")
    return example


@router.post("/examples", response_model=TrainingExampleResponse)
def create_example(example: TrainingExampleCreate, db: Session = Depends(get_db)):
    db_example = TrainingExample(**example.model_dump())
    db.add(db_example)
    db.commit()
    db.refresh(db_example)
    return db_example


@router.patch("/examples/{example_id}", response_model=TrainingExampleResponse)
def update_example(example_id: int, update: TrainingExampleUpdate, db: Session = Depends(get_db)):
    example = db.query(TrainingExample).filter(TrainingExample.id == example_id).first()
    if not example:
        raise HTTPException(status_code=404, detail="Example not found")

    update_data = update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(example, field, value)

    db.commit()
    db.refresh(example)
    return example


@router.delete("/examples/{example_id}")
def delete_example(example_id: int, db: Session = Depends(get_db)):
    example = db.query(TrainingExample).filter(TrainingExample.id == example_id).first()
    if not example:
        raise HTTPException(status_code=404, detail="Example not found")

    db.delete(example)
    db.commit()
    return {"message": "Deleted"}
