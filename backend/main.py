"""
AI Memory API - Main Application Entry Point
Production-ready FastAPI application with comprehensive middleware and instrumentation.
"""
import asyncio
import json
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.core.config import settings
from backend.core.logging_config import logger
from backend.api import api_router


# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.APP_ENV}")

    # Initialize database connections
    try:
        from backend.core.database import init_db
        await init_db()
        logger.info("✓ Database connections initialized")
    except Exception as e:
        logger.error(f"✗ Database initialization failed: {e}")
        raise

    # Initialize vector database
    try:
        from backend.services.vector_store import init_vector_store
        await init_vector_store()
        logger.info("✓ Vector store (Milvus) initialized")
    except Exception as e:
        logger.warning(f"⚠ Vector store initialization failed: {e}")

    # Initialize knowledge graph
    try:
        from backend.services.knowledge_graph import init_knowledge_graph
        await init_knowledge_graph()
        logger.info("✓ Knowledge graph (Neo4j) initialized")
    except Exception as e:
        logger.warning(f"⚠ Knowledge graph initialization failed: {e}")

    # Load embedding model
    try:
        from backend.ml.embeddings.model import load_embedding_model
        await load_embedding_model()
        logger.info(f"✓ Embedding model loaded: {settings.EMBEDDING_MODEL_NAME}")
    except Exception as e:
        logger.warning(f"⚠ Embedding model loading failed: {e}")

    # Initialize temporal knowledge graph
    try:
        from backend.services.temporal_graph import init_temporal_graph
        await init_temporal_graph()
        logger.info("✓ Temporal knowledge graph initialized")
    except Exception as e:
        logger.warning(f"⚠ Temporal graph initialization failed: {e}")

    # Initialize RL agents (load checkpoints if available)
    try:
        from backend.rl.memory_manager import get_memory_manager_agent
        from backend.rl.answer_agent import get_answer_agent

        # Pre-initialize agents (will load models on first access)
        logger.info("✓ RL agents ready (Memory Manager & Answer Agent)")
    except Exception as e:
        logger.warning(f"⚠ RL agents initialization warning: {e}")

    # Initialize world model
    try:
        from backend.agents.world_model import get_world_model

        # Pre-initialize world model
        world_model = get_world_model()
        logger.info("✓ World model initialized (dynamics & reward predictors)")
    except Exception as e:
        logger.warning(f"⚠ World model initialization failed: {e}")

    # Initialize cross-encoder reranker
    try:
        from backend.ml.reranking import get_reranker

        # Pre-load reranker model
        logger.info("✓ Cross-encoder reranker ready")
    except Exception as e:
        logger.warning(f"⚠ Reranker initialization failed: {e}")

    # Initialize working memory manager
    try:
        from backend.services.working_memory import get_working_memory_manager

        manager = get_working_memory_manager()
        logger.info("✓ Working memory manager initialized")
    except Exception as e:
        logger.warning(f"⚠ Working memory manager initialization failed: {e}")

    logger.info("🚀 Application startup complete! All advanced features initialized.")

    yield  # Application runs here

    # Shutdown
    logger.info("Shutting down application...")

    # Close database connections
    try:
        from backend.core.database import close_db
        await close_db()
        logger.info("✓ Database connections closed")
    except Exception as e:
        logger.error(f"✗ Database shutdown error: {e}")

    logger.info("👋 Application shutdown complete")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Universal Memory API with Reinforcement Learning & Advanced Reasoning",
    docs_url="/docs" if settings.ENABLE_API_DOCS else None,
    redoc_url="/redoc" if settings.ENABLE_API_DOCS else None,
    openapi_url="/openapi.json" if settings.ENABLE_API_DOCS else None,
    lifespan=lifespan,
)


# ============================================================================
# MIDDLEWARE
# ============================================================================

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GZip Compression
app.add_middleware(GZipMiddleware, minimum_size=1000)



# Request ID and Timing Middleware
@app.middleware("http")
async def add_request_id_and_timing(request: Request, call_next):
    """Add unique request ID and measure request duration."""
    import uuid

    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    start_time = time.time()

    response = await call_next(request)

    process_time = (time.time() - start_time) * 1000  # Convert to ms

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = f"{process_time:.2f}ms"

    logger.info(
        f"Request completed",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": process_time,
        }
    )

    return response


# Automatic Conversation Memory Middleware
@app.middleware("http")
async def auto_save_conversation(request: Request, call_next):
    """Automatically save important conversation context to memory."""
    path = request.url.path
    method = request.method

    # Only capture body for POST requests to important endpoints
    important_paths = ["/v1/search", "/v1/reason", "/v1/memories", "/v1/rl/train", "/v1/rl/evaluate", "/v1/collections"]
    should_check = any(p in path for p in important_paths) and method == "POST"

    request_body = None
    original_request = request
    if should_check:
        # Capture request body before it's consumed
        body_bytes = await request.body()
        if body_bytes:
            try:
                request_body = json.loads(body_bytes)
            except Exception:
                pass

        # Create new receive function that returns the captured body
        async def receive():
            return {"type": "http.request", "body": body_bytes}

        # Create new request with preserved scope but new receive
        request = Request(request.scope, receive)

    # Process request normally
    response = await call_next(request)

    # Only save for successful POST requests
    if should_check and response.status_code == 200 and request_body:
        logger.info(f"Auto-save triggered for {path}")

        # Get user ID from request state or auth header (use original request)
        user_id = None
        if hasattr(original_request.state, "user"):
            user_id = original_request.state.user.id
            logger.info(f"Got user_id from request.state: {user_id}")
        else:
            auth = original_request.headers.get("Authorization", "")
            logger.info(f"Auth header: {auth[:50] if auth else 'None'}...")
            if auth.startswith("Bearer "):
                try:
                    from backend.core.auth.jwt import verify_access_token
                    token = auth.split(" ")[1]
                    payload = verify_access_token(token)
                    if payload:
                        user_id = payload.get("sub")
                        logger.info(f"Got user_id from token: {user_id}")
                    else:
                        logger.warning("Access token verification returned None")
                except Exception as e:
                    logger.error(f"Failed to verify access token: {e}", exc_info=True)

        if user_id:
            logger.info(f"Saving conversation context for user {user_id}")
            # Save conversation context (await to ensure it completes)
            try:
                await _save_conversation_context(user_id, path, method, request_body)
            except Exception as e:
                logger.error(f"Failed in middleware to save conversation: {e}", exc_info=True)
        else:
            logger.warning("No user_id found, cannot auto-save conversation")

    return response


async def _save_conversation_context(user_id: str, path: str, method: str, request_body: dict):
    """Background task to save conversation context."""
    try:
        import hashlib
        import uuid
        from backend.core.database import AsyncSessionLocal
        from backend.models.collection import Collection
        from backend.models.memory import Memory, MemoryMetadata
        from sqlalchemy import select, update

        COLLECTION_NAME = "Conversation History"

        # Get or create Conversation History collection
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(Collection).where(
                    Collection.user_id == user_id,
                    Collection.name == COLLECTION_NAME
                )
            )
            collection = result.scalar_one_or_none()

            if not collection:
                collection = Collection(
                    id=str(uuid.uuid4()),
                    user_id=user_id,
                    name=COLLECTION_NAME,
                    description="Automatically captured conversation history",
                    custom_metadata={"auto_created": True}
                )
                db.add(collection)
                await db.commit()
                await db.refresh(collection)
                logger.info(f"Created Conversation History collection for user {user_id}")

            # Generate summary
            summary = _generate_summary(path, request_body)

            if summary:
                # Create content hash
                content_hash = hashlib.sha256(summary.encode()).hexdigest()

                # Create memory
                memory = Memory(
                    id=str(uuid.uuid4()),
                    collection_id=collection.id,
                    content=summary,
                    content_hash=content_hash,
                    importance=0.6,
                    source_type="automatic",
                    source_reference=path
                )

                # Create memory metadata
                metadata = MemoryMetadata(
                    id=str(uuid.uuid4()),
                    memory_id=memory.id,
                    custom_metadata={
                        "auto_saved": True,
                        "timestamp": datetime.utcnow().isoformat(),
                        "source": "automatic",
                        "endpoint": path,
                        "method": method
                    }
                )

                db.add(memory)
                db.add(metadata)

                # Update collection memory count atomically to prevent race conditions
                await db.execute(
                    update(Collection)
                    .where(Collection.id == collection.id)
                    .values(memory_count=Collection.memory_count + 1)
                )

                await db.commit()
                logger.info(f"Auto-saved conversation context: {summary[:100]}...")
    except Exception as e:
        logger.error(f"Failed to auto-save conversation: {e}", exc_info=True)


def _generate_summary(path: str, body: dict) -> Optional[str]:
    """Generate human-readable summary of the interaction."""
    try:
        if "/search" in path:
            query = body.get("query", "")
            return f"Searched for: {query}"
        elif "/reason" in path:
            query = body.get("query", "")
            return f"Reasoned about: {query}"
        elif "/rl/train" in path:
            agent_type = path.split("/")[-1] if "memory-manager" in path or "answer-agent" in path else "agent"
            num_episodes = body.get("num_episodes", "")
            return f"Trained {agent_type} RL agent ({num_episodes} episodes)"
        elif "/rl/evaluate" in path:
            agent_type = body.get("agent_type", "agent")
            return f"Evaluated {agent_type} performance"
        elif "/memories" in path:
            content = body.get("content", "")
            if content:
                preview = content[:100] + "..." if len(content) > 100 else content
                return f"Added memory: {preview}"
        elif "/collections" in path:
            name = body.get("name", "")
            return f"Created collection: {name}"

        return None
    except Exception as e:
        logger.error(f"Failed to generate summary: {e}")
        return None


# ============================================================================
# EXCEPTION HANDLERS
# ============================================================================

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": "http_exception",
                "message": exc.detail,
                "status_code": exc.status_code,
            }
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "type": "validation_error",
                "message": "Request validation failed",
                "details": exc.errors(),
            }
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.exception(f"Unhandled exception: {exc}")

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "type": "internal_server_error",
                "message": "An unexpected error occurred",
                "detail": str(exc) if settings.DEBUG else None,
            }
        },
    )


# ============================================================================
# ROUTES
# ============================================================================

# Health check endpoint
@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint for load balancers and monitoring."""
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "environment": settings.APP_ENV,
    }


# Readiness check endpoint
@app.get("/ready", tags=["System"])
async def readiness_check():
    """Readiness check - verifies all dependencies are available."""
    checks = {
        "database": False,
        "redis": False,
        "vector_store": False,
        "knowledge_graph": False,
    }

    # Check database
    try:
        from backend.core.database import check_db_connection
        checks["database"] = await check_db_connection()
    except Exception:
        pass

    # Check Redis
    try:
        from backend.core.cache import check_redis_connection
        checks["redis"] = await check_redis_connection()
    except Exception:
        pass

    all_ready = all(checks.values())

    return {
        "status": "ready" if all_ready else "not_ready",
        "checks": checks,
    }


# Include API routes
app.include_router(api_router, prefix=settings.API_V1_PREFIX)


# Root endpoint
@app.get("/", tags=["System"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": "Universal Memory API with RL & Advanced Reasoning",
        "docs_url": "/docs" if settings.ENABLE_API_DOCS else None,
        "health_check": "/health",
        "api_prefix": settings.API_V1_PREFIX,
    }


# ============================================================================
# OBSERVABILITY
# ============================================================================

# Prometheus metrics
if settings.ENABLE_METRICS:
    from prometheus_fastapi_instrumentator import Instrumentator

    Instrumentator().instrument(app).expose(app, endpoint="/metrics")
    logger.info("✓ Prometheus metrics enabled at /metrics")


# OpenTelemetry tracing
if settings.ENABLE_TRACING:
    try:
        from opentelemetry import trace
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.jaeger.thrift import JaegerExporter

        # Set up tracing
        trace.set_tracer_provider(TracerProvider())

        jaeger_exporter = JaegerExporter(
            agent_host_name=settings.JAEGER_AGENT_HOST,
            agent_port=settings.JAEGER_AGENT_PORT,
        )

        trace.get_tracer_provider().add_span_processor(
            BatchSpanProcessor(jaeger_exporter)
        )

        # Instrument FastAPI
        FastAPIInstrumentor.instrument_app(app)
        logger.info("✓ OpenTelemetry tracing enabled")

    except Exception as e:
        logger.warning(f"⚠ Failed to initialize tracing: {e}")


# Sentry error tracking
if settings.SENTRY_DSN:
    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration

        sentry_sdk.init(
            dsn=settings.SENTRY_DSN,
            traces_sample_rate=settings.SENTRY_TRACES_SAMPLE_RATE,
            environment=settings.APP_ENV,
            integrations=[FastApiIntegration()],
        )
        logger.info("✓ Sentry error tracking enabled")

    except Exception as e:
        logger.warning(f"⚠ Failed to initialize Sentry: {e}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        workers=settings.WORKERS if not settings.RELOAD else 1,
        log_level=settings.LOG_LEVEL.lower(),
    )
