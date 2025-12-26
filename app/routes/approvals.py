from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from ..database import get_db
from ..models.approval import Approval
from ..schemas.approval import ApprovalCreate, ApprovalUpdate, ApprovalResponse

router = APIRouter(prefix="/api/approvals", tags=["approvals"])


@router.get("", response_model=List[ApprovalResponse])
def get_approvals(status: str = None, db: Session = Depends(get_db)):
    query = db.query(Approval)
    if status:
        query = query.filter(Approval.status == status)
    return query.order_by(Approval.created_at.desc()).all()


@router.get("/{approval_id}", response_model=ApprovalResponse)
def get_approval(approval_id: int, db: Session = Depends(get_db)):
    approval = db.query(Approval).filter(Approval.id == approval_id).first()
    if not approval:
        raise HTTPException(status_code=404, detail="Approval not found")
    return approval


@router.post("", response_model=ApprovalResponse)
def create_approval(approval: ApprovalCreate, db: Session = Depends(get_db)):
    db_approval = Approval(**approval.model_dump())
    db.add(db_approval)
    db.commit()
    db.refresh(db_approval)
    return db_approval


@router.patch("/{approval_id}", response_model=ApprovalResponse)
def update_approval(approval_id: int, update: ApprovalUpdate, db: Session = Depends(get_db)):
    approval = db.query(Approval).filter(Approval.id == approval_id).first()
    if not approval:
        raise HTTPException(status_code=404, detail="Approval not found")

    approval.status = update.status
    db.commit()
    db.refresh(approval)
    return approval


@router.post("/{approval_id}/approve", response_model=ApprovalResponse)
def approve(approval_id: int, db: Session = Depends(get_db)):
    approval = db.query(Approval).filter(Approval.id == approval_id).first()
    if not approval:
        raise HTTPException(status_code=404, detail="Approval not found")

    approval.status = "approved"
    db.commit()
    db.refresh(approval)
    return approval


@router.post("/{approval_id}/reject", response_model=ApprovalResponse)
def reject(approval_id: int, db: Session = Depends(get_db)):
    approval = db.query(Approval).filter(Approval.id == approval_id).first()
    if not approval:
        raise HTTPException(status_code=404, detail="Approval not found")

    approval.status = "rejected"
    db.commit()
    db.refresh(approval)
    return approval
