"""CloudFinOps Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import CloudFinOpsAction, CloudFinOpsObservation, ServerState
except ImportError:
    from models import CloudFinOpsAction, CloudFinOpsObservation, ServerState


class CloudFinOpsEnv(
    EnvClient[CloudFinOpsAction, CloudFinOpsObservation, State]
):
    """
    Client for the CloudFinOps Environment.

    Maintains a persistent WebSocket connection to the environment server.

    Example:
        >>> with CloudFinOpsEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.budget_remaining)
        ...
        ...     result = client.step(CloudFinOpsAction(command="TERMINATE", target_id="idle-0"))
        ...     print(result.observation.budget_remaining)

    Example with Docker:
        >>> client = CloudFinOpsEnv.from_docker_image("cloudfinops_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(CloudFinOpsAction(command="IGNORE"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: CloudFinOpsAction) -> Dict:
        """Convert CloudFinOpsAction to JSON payload for step message."""
        return {
            "command": action.command,
            "target_id": action.target_id,
            "reply": action.reply or "",
        }

    def _parse_result(self, payload: Dict) -> StepResult[CloudFinOpsObservation]:
        """Parse server response into StepResult[CloudFinOpsObservation]."""
        obs_data = payload.get("observation", {})

        # Parse servers
        servers = []
        for s in obs_data.get("servers", []):
            servers.append(ServerState(**s))

        observation = CloudFinOpsObservation(
            servers=servers,
            traffic_load=obs_data.get("traffic_load", 0.0),
            spike_detected=obs_data.get("spike_detected", False),
            incidents=obs_data.get("incidents", []),
            budget_remaining=obs_data.get("budget_remaining", 0.0),
            time_step=obs_data.get("time_step", 0),
            inbox=obs_data.get("inbox", []),
            carbon_kwh=obs_data.get("carbon_kwh", 0.0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
