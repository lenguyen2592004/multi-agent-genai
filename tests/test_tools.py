from pathlib import Path

from data.init_db import initialize_sqlite
from rag.pipeline import RAGPipeline
from tools.python_executor import PythonExecutorTool
from tools.sqlite_tool import SQLiteQueryTool


def test_sqlite_tool_allows_select(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    initialize_sqlite(db_path)
    tool = SQLiteQueryTool(db_path=db_path)

    result = tool.run({"query": "SELECT name, owner FROM tasks LIMIT 2;"})
    assert result["status"] == "success"
    assert isinstance(result["result"], list)


def test_sqlite_tool_blocks_mutation(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    initialize_sqlite(db_path)
    tool = SQLiteQueryTool(db_path=db_path)

    result = tool.run({"query": "DROP TABLE tasks;"})
    assert result["status"] == "error"


def test_python_executor_blocks_unsafe_code() -> None:
    tool = PythonExecutorTool(timeout_seconds=1)
    result = tool.run({"code": "import os\nprint('x')"})
    assert result["status"] == "error"


def test_rag_ingest_and_search(tmp_path: Path) -> None:
    vector_path = tmp_path / "vectors.json"
    rag = RAGPipeline(vector_store_path=vector_path)
    rag.ingest_text(
        document_id="doc1",
        text="The project must assign one owner for each action item.",
        metadata={"source": "unit-test"},
    )

    docs = rag.search("Who owns the action item?", top_k=2)
    assert len(docs) >= 1
