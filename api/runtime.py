from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from agents.orchestrator import AgentOrchestrator
from data.init_db import initialize_sqlite
from llm.ollama_client import OllamaClient
from observability.metrics import MetricsStore
from rag.pipeline import RAGPipeline
from tools.registry import ToolRegistry


@dataclass
class RuntimeServices:
    orchestrator: AgentOrchestrator
    rag_pipeline: RAGPipeline
    metrics: MetricsStore
    db_path: Path


@lru_cache(maxsize=1)
def get_runtime_services() -> RuntimeServices:
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data"
    docs_dir = data_dir / "docs"
    db_path = data_dir / "app.db"
    vector_store_path = data_dir / "vector_store.json"

    data_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    initialize_sqlite(db_path)

    rag_pipeline = RAGPipeline(vector_store_path=vector_store_path)
    if rag_pipeline.document_count() == 0:
        rag_pipeline.ingest_directory(docs_dir)

    tool_registry = ToolRegistry(rag_pipeline=rag_pipeline, db_path=db_path)
    llm_client = OllamaClient()
    orchestrator = AgentOrchestrator(
        llm_client=llm_client,
        rag_pipeline=rag_pipeline,
        tool_registry=tool_registry,
    )
    metrics = MetricsStore()

    return RuntimeServices(
        orchestrator=orchestrator,
        rag_pipeline=rag_pipeline,
        metrics=metrics,
        db_path=db_path,
    )
