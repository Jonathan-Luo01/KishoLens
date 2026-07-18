import os
from typing import Optional
from sqlmodel import SQLModel, Field, create_engine

# Database configuration
DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "kisholens.db"
)

class Novel(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    author: str
    source: str
    genre: Optional[str] = Field(default=None)
    territory: Optional[str] = Field(default=None)

class Chapter(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    id: Optional[int] = Field(default=None, primary_key=True)
    novel_id: int = Field(foreign_key="novel.id")
    chapter_number: int
    title: str
    text_ja: str
    text_en: str
    text_zh: str = Field(default="")

def get_engine(db_path: str = DEFAULT_DB_PATH):
    """Creates and returns the SQLite engine."""
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    return create_engine(f"sqlite:///{db_path}")
