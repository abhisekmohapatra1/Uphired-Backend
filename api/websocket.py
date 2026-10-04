import asyncio
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from api.routes import active_workflows

ws_router = APIRouter()


@ws_router.websocket("/ws/execution/{workflow_id}")
async def execution_ws(websocket: WebSocket, workflow_id: str):
    """
    WebSocket endpoint — streams live execution logs to the frontend.
    Client connects immediately after starting a workflow.
    Server pushes log updates every 500ms until workflow completes.
    """
    await websocket.accept()

    try:
        last_log_count = 0

        while True:
            workflow = active_workflows.get(workflow_id)

            if not workflow:
                await websocket.send_json({"type": "error", "message": "Workflow not found"})
                break

            logs = workflow.get("logs", [])

            # Only push new log lines
            if len(logs) > last_log_count:
                new_logs = logs[last_log_count:]
                for log_line in new_logs:
                    await websocket.send_json({
                        "type": "log",
                        "message": log_line,
                        "node": workflow.get("current_node", ""),
                    })
                last_log_count = len(logs)

            # Notify completion
            if workflow["status"] in ("done", "error"):
                await websocket.send_json({
                    "type": "complete",
                    "status": workflow["status"],
                    "result": workflow.get("result"),
                })
                break

            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        pass