import os
import json
import hashlib
from typing import Dict, Optional
import logging

from fastapi import APIRouter, Request, BackgroundTasks, HTTPException, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from integrations.github_integration import (
    verify_github_signature,
    handle_pull_request_event as handle_github_pr
)
from integrations.gitlab_integration import (
    handle_merge_request_event as handle_gitlab_mr
)

logger = logging.getLogger(__name__)

# create router
router = APIRouter(prefix="/webhook", tags=["webhooks"])

# track processed events to ensure idempotency
_processed_events = set()
MAX_PROCESSED_EVENTS = 1000  # Limit memory usage


class WebhookResponse(BaseModel):
    """Standard webhook response."""
    status: str
    message: str
    event_id: Optional[str] = None


def get_event_id(provider: str, payload: Dict) -> str:
    """
    Generate a unique event ID for idempotency.
    
    Args:
        provider: 'github' or 'gitlab'
        payload: Webhook payload
        
    Returns:
        Unique event ID
    """
    if provider == "github":
        # gitHub provides delivery ID in headers, but we'll use payload data
        pr = payload.get("pull_request", {})
        repo = payload.get("repository", {})
        action = payload.get("action", "")
        
        # include commit SHA for synchronize events
        head_sha = pr.get("head", {}).get("sha", "")
        
        key_parts = [
            repo.get("full_name", ""),
            str(pr.get("number", "")),
            action,
            head_sha[:8] if head_sha else ""
        ]
    elif provider == "gitlab":
        # gitLab webhook event ID
        obj_attr = payload.get("object_attributes", {})
        project = payload.get("project", {})
        
        key_parts = [
            str(project.get("id", "")),
            str(obj_attr.get("iid", "")),
            obj_attr.get("action", ""),
            obj_attr.get("last_commit", {}).get("id", "")[:8]
        ]
    else:
        # fallback: hash the entire payload
        return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    
    # create hash from key parts
    key = "-".join(filter(None, key_parts))
    return hashlib.md5(key.encode()).hexdigest()[:16]


def is_duplicate_event(event_id: str) -> bool:
    """
    Check if an event has already been processed.
    
    Args:
        event_id: Unique event identifier
        
    Returns:
        True if duplicate, False otherwise
    """
    if event_id in _processed_events:
        return True
    
    # add to processed set
    _processed_events.add(event_id)
    
    # cleanup old events if set gets too large
    if len(_processed_events) > MAX_PROCESSED_EVENTS:
        # remove oldest 20%
        to_remove = int(MAX_PROCESSED_EVENTS * 0.2)
        for _ in range(to_remove):
            _processed_events.pop()
    
    return False


@router.post("/github", response_model=WebhookResponse)
async def github_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_hub_signature_256: Optional[str] = Header(None),
    x_github_event: Optional[str] = Header(None),
    x_github_delivery: Optional[str] = Header(None)
):
    """
    Handle GitHub webhook events.
    
    Processes pull request events and triggers code quality analysis.
    """
    # get request body
    body = await request.body()
    
    # verify signature
    if not verify_github_signature(body, x_hub_signature_256 or ""):
        logger.warning("Invalid GitHub webhook signature")
        raise HTTPException(status_code=401, detail="Invalid signature")
    
    # parse payload
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    
    # check event type
    if x_github_event != "pull_request":
        return WebhookResponse(
            status="ignored",
            message=f"Event type '{x_github_event}' not processed"
        )
    
    # generate event ID for idempotency
    event_id = get_event_id("github", payload)
    
    # check for duplicate
    if is_duplicate_event(event_id):
        logger.info(f"Duplicate GitHub event {event_id}, skipping")
        return WebhookResponse(
            status="duplicate",
            message="Event already processed",
            event_id=event_id
        )
    
    # log event
    action = payload.get("action", "unknown")
    pr_number = payload.get("pull_request", {}).get("number", "unknown")
    repo_name = payload.get("repository", {}).get("full_name", "unknown")
    logger.info(f"GitHub PR event: {repo_name}#{pr_number} - {action}")
    
    # queue background task for analysis
    background_tasks.add_task(
        process_github_event,
        payload,
        event_id
    )
    
    return WebhookResponse(
        status="accepted",
        message="Pull request analysis queued",
        event_id=event_id
    )


@router.post("/gitlab", response_model=WebhookResponse)
async def gitlab_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_gitlab_event: Optional[str] = Header(None),
    x_gitlab_token: Optional[str] = Header(None)
):
    """
    Handle GitLab webhook events.
    
    Processes merge request events and triggers code quality analysis.
    """
    # verify token if configured
    expected_token = os.getenv("GITLAB_WEBHOOK_TOKEN")
    if expected_token and x_gitlab_token != expected_token:
        logger.warning("Invalid GitLab webhook token")
        raise HTTPException(status_code=401, detail="Invalid token")
    
    # parse payload
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    
    # check event type
    if x_gitlab_event not in ["Merge Request Hook", "merge_request"]:
        return WebhookResponse(
            status="ignored",
            message=f"Event type '{x_gitlab_event}' not processed"
        )
    
    # generate event ID
    event_id = get_event_id("gitlab", payload)
    
    # check for duplicate
    if is_duplicate_event(event_id):
        logger.info(f"Duplicate GitLab event {event_id}, skipping")
        return WebhookResponse(
            status="duplicate",
            message="Event already processed",
            event_id=event_id
        )
    
    # log event
    obj_attr = payload.get("object_attributes", {})
    action = obj_attr.get("action", "unknown")
    mr_iid = obj_attr.get("iid", "unknown")
    project_path = payload.get("project", {}).get("path_with_namespace", "unknown")
    logger.info(f"GitLab MR event: {project_path}!{mr_iid} - {action}")
    
    # queue background task
    background_tasks.add_task(
        process_gitlab_event,
        payload,
        event_id
    )
    
    return WebhookResponse(
        status="accepted",
        message="Merge request analysis queued",
        event_id=event_id
    )


async def process_github_event(payload: Dict, event_id: str):
    """
    Process GitHub pull request event in background.
    
    Args:
        payload: GitHub webhook payload
        event_id: Unique event identifier
    """
    try:
        logger.info(f"Processing GitHub event {event_id}")
        result = await handle_github_pr(payload)
        logger.info(f"GitHub event {event_id} processed: {result}")
    except Exception as e:
        logger.error(f"Error processing GitHub event {event_id}: {e}")


async def process_gitlab_event(payload: Dict, event_id: str):
    """
    Process GitLab merge request event in background.
    
    Args:
        payload: GitLab webhook payload
        event_id: Unique event identifier
    """
    try:
        logger.info(f"Processing GitLab event {event_id}")
        result = await handle_gitlab_mr(payload)
        logger.info(f"GitLab event {event_id} processed: {result}")
    except Exception as e:
        logger.error(f"Error processing GitLab event {event_id}: {e}")


# health check endpoint for webhooks
@router.get("/health")
async def webhook_health():
    """Check webhook endpoint health."""
    return {
        "status": "healthy",
        "processed_events": len(_processed_events),
        "github_configured": bool(os.getenv("GITHUB_TOKEN")),
        "gitlab_configured": bool(os.getenv("GITLAB_TOKEN"))
    }


# app.include_router(webhook_router)