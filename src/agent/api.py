"""
FastAPI Orchestrator
Primary Responsibility: Exposes the HTTP endpoints for the admission assistant.
"""
import uuid
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from opentelemetry import trace

from src.agent.retriever import FAISSRetriever
from src.agent.generator import LocalQwenGenerator
from src.agent.cutoffs import CutoffLookup
from src.agentops.telemetry import tracer, tracker
from src.config import AGENTOPS_ENABLED

app = FastAPI(title="AgentOps Admission API")

retriever = FAISSRetriever()
generator = LocalQwenGenerator()
cutoffs = CutoffLookup()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    trace_id: str
    answer: str
    context: list

@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    trace_id = str(uuid.uuid4())
    trace_start = datetime.now(timezone.utc)
    
    with tracer.start_as_current_span("admission.query") as root_span:
        root_span.set_attribute("trace_id", trace_id)
        root_span.set_attribute("agentops_enabled", AGENTOPS_ENABLED)
        
        with tracer.start_as_current_span("query.validation") as val_span:
            val_span.set_attribute("query_length", len(request.query))
            if not request.query.strip():
                val_span.set_attribute("classification", "EMPTY_QUERY")
                val_span.set_status(trace.Status(trace.StatusCode.ERROR))
                root_span.set_status(trace.Status(trace.StatusCode.ERROR))
                raise HTTPException(status_code=400, detail="Empty query")
            val_span.set_attribute("classification", "VALID")
            
        try:
            # 1. Retrieve Context
            context = retriever.retrieve(query=request.query, trace_id=trace_id)

            # Rank/cutoff questions also get exact rows from the structured JoSAA table
            cutoff_context = cutoffs.lookup(request.query)
            if cutoff_context:
                context = [cutoff_context] + context

            # 2. Prompt Construction is embedded in generator, but we can capture it here virtually or in generator.
            # 3. Generate Answer
            answer = await generator.generate_response(
                query=request.query, 
                context=context, 
                trace_id=trace_id
            )
            
            # 4. Grounding Validation
            with tracer.start_as_current_span("grounding.validation") as ground_span:
                if not context:
                    ground_span.set_attribute("support_status", "UNSUPPORTED")
                    ground_span.set_status(trace.Status(trace.StatusCode.ERROR))
                else:
                    ground_span.set_attribute("support_status", "SUPPORTED")
                    ground_span.set_attribute("evidence_score", context[0]['distance'])
            
            # 5. Explanation
            with tracer.start_as_current_span("explanation") as exp_span:
                exp_span.set_attribute("evidence_count", len(context))
                exp_span.set_attribute("confidence_indicator", "High" if context else "Low")
                
            # 6. Response
            with tracer.start_as_current_span("response") as res_span:
                res_span.set_attribute("response_status", "success")
                res_span.set_attribute("abstained", False)

            # Keeping MySQL fallback for now
            tracker.emit_span(
                trace_id=trace_id,
                span_type="full_query_lifecycle",
                start_time=trace_start,
                end_time=datetime.now(timezone.utc),
                status="success",
                metadata={"query": request.query}
            )
            
            return QueryResponse(
                trace_id=trace_id,
                answer=answer,
                context=context 
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()

            root_span.set_status(trace.Status(trace.StatusCode.ERROR))
            tracker.emit_span(
                trace_id=trace_id,
                span_type="full_query_lifecycle",
                start_time=trace_start,
                end_time=datetime.now(timezone.utc),
                status="failure",
                error_msg=str(e),
                metadata={"query": request.query}
            )
            raise HTTPException(status_code=500, detail="Internal Server Error during processing.")
