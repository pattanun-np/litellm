"""
Transformation utilities for Anthropic Batches
- Build Anthropic batch requests from JSONL
- Map Anthropic batch responses to LiteLLMBatch
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.types.utils import LiteLLMBatch


def iso_to_epoch(iso_str: Optional[str]) -> Optional[int]:
    if not iso_str:
        return None
    try:
        if iso_str.endswith("Z"):
            dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(iso_str)
        return int(dt.timestamp())
    except Exception:
        return None

def build_requests_from_jsonl(jsonl_content: str) -> List[Dict[str, Any]]:
    """
    Parse OpenAI-style JSONL content into Anthropic batch requests.
    Each line may be either {"body": {...}} or the params object itself.
    """
    requests: List[Dict[str, Any]] = []
    if not jsonl_content:
        return requests
    lines = jsonl_content.splitlines()
    anthropic_config = AnthropicConfig()
    for idx, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            # Skip invalid JSON lines
            continue

        # Support {"body": {...}} shape
        body = obj.get("body") if isinstance(obj, dict) else None
        if body is None and isinstance(obj, dict):
            body = obj
        if not isinstance(body, dict):
            continue

        # Extract model/messages and non-default (OpenAI-style) params
        model = body.get("model")
        messages = body.get("messages", [])
        non_default_params = {k: v for k, v in body.items() if k not in ("model", "messages")}

        # Map OpenAI-style params to Anthropic optional params (drops unsupported like 'strict')
        optional_params: Dict[str, Any] = {}
        optional_params = anthropic_config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=True,
        )
        
        # json_mode is not supported for batches
        optional_params.pop("json_mode", False)

        # Transform to Anthropic params
        headers: Dict[str, str] = {}
        litellm_params: Dict[str, Any] = {}
        anthropic_params = anthropic_config.transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )
        custom_id = obj.get("custom_id") or f"req-{idx+1}"
        requests.append({
            "custom_id": custom_id,
            "params": anthropic_params,
        })
    return requests


def to_litellm_batch(batch_obj: Dict[str, Any]) -> LiteLLMBatch:
    """Map Anthropic message_batch to LiteLLMBatch."""
    processing_status = batch_obj.get("processing_status")
    created_at_iso = batch_obj.get("created_at")
    ended_at_iso = batch_obj.get("ended_at")
    cancel_initiated_at_iso = batch_obj.get("cancel_initiated_at")

    # Map request_counts to expected schema: completed, failed, total
    anth_counts: Dict[str, Any] = batch_obj.get("request_counts", {}) or {}
    processing = int(anth_counts.get("processing", 0) or 0)
    succeeded = int(anth_counts.get("succeeded", 0) or 0)
    errored = int(anth_counts.get("errored", 0) or 0)
    canceled = int(anth_counts.get("canceled", 0) or 0)
    expired = int(anth_counts.get("expired", 0) or 0)

    completed = succeeded
    failed = errored + canceled + expired
    total = processing + succeeded + errored + canceled + expired
    request_counts_mapped: Dict[str, int] = {
        "completed": completed,
        "failed": failed,
        "total": total,
    }

    # Derive OpenAI-compatible status
    if processing_status == "canceling":
        status = "cancelling"
    elif processing_status == "in_progress":
        status = "in_progress"
    elif processing_status == "ended":
        if canceled > 0 and succeeded == 0 and errored == 0 and expired == 0:
            status = "cancelled"
        elif expired > 0 and succeeded == 0 and errored == 0 and canceled == 0:
            status = "expired"
        elif errored > 0 and succeeded == 0:
            status = "failed"
        elif errored > 0 or canceled > 0 or expired > 0:
            # Mixed outcomes -> treat as failed
            status = "failed"
        else:
            status = "completed"
    else:
        status = processing_status or "validating"

    created_at_epoch = iso_to_epoch(created_at_iso) or int(datetime.now(tz=timezone.utc).timestamp())
    ended_at_epoch = iso_to_epoch(ended_at_iso)
    cancelling_at_epoch = iso_to_epoch(cancel_initiated_at_iso)

    # Status-specific timestamps
    completed_at_epoch = ended_at_epoch if status == "completed" else None
    cancelled_at_epoch = ended_at_epoch if status == "cancelled" else None
    expired_at_epoch = ended_at_epoch if status == "expired" else None
    failed_at_epoch = ended_at_epoch if status == "failed" else None

    return LiteLLMBatch(
        id=batch_obj.get("id", str(uuid.uuid4())),
        object="batch",
        endpoint=batch_obj.get("endpoint", "/v1/messages"),
        input_file_id="",  # must be a string per schema
        completion_window="24h",
        status=status,
        created_at=created_at_epoch,
        metadata=None,
        request_counts=request_counts_mapped,
        usage=None,
        errors=None,
        output_file_id=batch_obj.get("results_url") or None,
        error_file_id=None,
        expired_at=expired_at_epoch,
        expires_at=iso_to_epoch(batch_obj.get("expires_at")),
        failed_at=failed_at_epoch,
        finalizing_at=None,
        in_progress_at=None,
        cancelled_at=cancelled_at_epoch,
        cancelling_at=cancelling_at_epoch,
        completed_at=completed_at_epoch,
    )
