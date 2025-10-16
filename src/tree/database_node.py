from typing import List, Optional

from sqlalchemy import Boolean, Float, ForeignKey, Index, Integer, LargeBinary
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class DatabaseNode(Base):
    __tablename__ = "node"

    ## Columns
    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # Foreign key to parent node; nullable for root node
    parent_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("node.id"), nullable=True
    )

    # Predictive process input data
    model_input: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    # Scenario values data
    scenario: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)

    # Martingale values data
    martingale: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)

    # Node probability in the scenario tree
    probability: Mapped[float] = mapped_column(Float, nullable=False)

    # Check if the node is a leaf node
    is_leaf: Mapped[bool] = mapped_column(Boolean, nullable=False)

    # Relationships
    parent: Mapped[Optional["DatabaseNode"]] = relationship(
        "DatabaseNode", remote_side=[id], back_populates="children"
    )
    children: Mapped[List["DatabaseNode"]] = relationship(
        "DatabaseNode", back_populates="parent"
    )

    # Add index for common queries to retrieve children of a node and check leaf status
    __table_args__ = (Index("idx_parent_leaf", "parent_id", "is_leaf"),)
