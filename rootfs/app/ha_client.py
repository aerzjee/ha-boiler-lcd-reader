"""Home Assistant Supervisor API client.

Handles asynchronous communication with Home Assistant Core API via the Supervisor proxy,
using the injected SUPERVISOR_TOKEN for Zero-Trust authentication.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class HAClientError(Exception):
    """Exception raised for Home Assistant Supervisor API communication failures."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


# ---------------------------------------------------------------------------
# Client Implementation
# ---------------------------------------------------------------------------


class HomeAssistantClient:
    """Asynchronous client for Home Assistant Core API via Supervisor proxy."""

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        raw_url = base_url or os.getenv("HA_BASE_URL", "http://supervisor/core/api")
        self.base_url = raw_url.rstrip("/")
        self.token = token or os.getenv("SUPERVISOR_TOKEN")
        self.timeout_seconds = timeout_seconds
        self._client: httpx.AsyncClient | None = None

        if not self.token:
            logger.warning(
                "SUPERVISOR_TOKEN is not set in environment. "
                "API requests to Home Assistant will fail unless mock/local mode is configured."
            )

    @property
    def is_connected(self) -> bool:
        """Check if the internal HTTP client session is active."""
        return self._client is not None and not self._client.is_closed

    async def start(self) -> None:
        """Initialize the shared httpx.AsyncClient session."""
        if self._client is None or self._client.is_closed:
            headers = {"Accept": "application/json"}
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"

            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=httpx.Timeout(self.timeout_seconds, connect=5.0),
            )
            logger.debug("Initialized HomeAssistantClient session for %s", self.base_url)

    async def close(self) -> None:
        """Close the underlying httpx.AsyncClient session."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
            logger.debug("Closed HomeAssistantClient session")

    async def __aenter__(self) -> HomeAssistantClient:
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    def _ensure_client(self) -> httpx.AsyncClient:
        """Return the active HTTP client or raise if not started."""
        if self._client is None or self._client.is_closed:
            raise HAClientError("HomeAssistantClient is not started or has been closed.")
        return self._client

    async def get_camera_entities(self) -> list[dict[str, str]]:
        """Query Home Assistant for all available camera entities.

        Returns:
            List of camera entities with entity_id, friendly_name, and state,
            sorted alphabetically by entity_id.
        """
        client = self._ensure_client()
        try:
            response = await client.get("/states")
        except httpx.TimeoutException as exc:
            raise HAClientError("Timeout while querying Home Assistant states", status_code=504) from exc
        except httpx.RequestError as exc:
            raise HAClientError(f"Network error querying Home Assistant states: {exc}") from exc

        if response.status_code != 200:
            raise HAClientError(
                f"Failed to fetch states from Home Assistant: HTTP {response.status_code} - {response.text}",
                status_code=response.status_code,
            )

        try:
            states = response.json()
        except ValueError as exc:
            raise HAClientError("Invalid JSON response from Home Assistant /states endpoint") from exc

        cameras: list[dict[str, str]] = []
        for state_obj in states:
            entity_id = state_obj.get("entity_id", "")
            if entity_id.startswith("camera."):
                attributes = state_obj.get("attributes", {})
                friendly_name = attributes.get("friendly_name", entity_id)
                current_state = str(state_obj.get("state", "unknown"))
                cameras.append(
                    {
                        "entity_id": entity_id,
                        "friendly_name": friendly_name,
                        "state": current_state,
                    }
                )

        cameras.sort(key=lambda c: c["entity_id"])
        return cameras

    async def get_camera_snapshot(self, entity_id: str) -> bytes:
        """Fetch a JPEG camera snapshot for the specified camera entity.

        Args:
            entity_id: The camera entity ID (must start with 'camera.')

        Returns:
            Raw JPEG bytes of the camera snapshot.

        Raises:
            HAClientError: On invalid entity ID, HTTP errors, or timeouts.
        """
        if not entity_id or not entity_id.startswith("camera."):
            raise HAClientError(
                f"Invalid camera entity ID '{entity_id}'. Must start with 'camera.'",
                status_code=400,
            )

        client = self._ensure_client()
        url = f"/camera_proxy/{entity_id}"
        try:
            response = await client.get(url)
        except httpx.TimeoutException as exc:
            raise HAClientError(
                f"Timeout fetching camera snapshot for '{entity_id}'",
                status_code=504,
            ) from exc
        except httpx.RequestError as exc:
            raise HAClientError(f"Network error fetching camera snapshot for '{entity_id}': {exc}") from exc

        if response.status_code != 200:
            raise HAClientError(
                f"Failed to fetch snapshot for '{entity_id}': HTTP {response.status_code} - {response.text}",
                status_code=response.status_code,
            )

        content = response.content
        if not content:
            raise HAClientError(f"Empty snapshot response received for camera '{entity_id}'", status_code=502)

        return content

    async def publish_temperature(
        self,
        entity_id: str,
        value: int,
        friendly_name: str,
        extra_attributes: dict[str, Any] | None = None,
    ) -> bool:
        """Publish an integer temperature sensor reading to Home Assistant.

        Temperatures are strictly integers with no decimal places.

        Args:
            entity_id: Target sensor entity ID (e.g. 'sensor.boiler_co_temperature')
            value: Integer temperature value in °C
            friendly_name: Friendly name for Home Assistant UI
            extra_attributes: Optional extra attributes (confidence, timestamp, etc.)

        Returns:
            True if the state was successfully published.

        Raises:
            HAClientError: On invalid entity, HTTP errors, or network failures.
        """
        if not entity_id or not entity_id.startswith("sensor."):
            raise HAClientError(
                f"Invalid sensor entity ID '{entity_id}'. Must start with 'sensor.'",
                status_code=400,
            )

        int_value = int(value)
        attributes: dict[str, Any] = {
            "unit_of_measurement": "°C",
            "device_class": "temperature",
            "state_class": "measurement",
            "friendly_name": friendly_name,
            "icon": "mdi:thermometer",
        }
        if extra_attributes:
            attributes.update(extra_attributes)

        payload = {
            "state": str(int_value),
            "attributes": attributes,
        }

        client = self._ensure_client()
        url = f"/states/{entity_id}"
        try:
            response = await client.post(url, json=payload)
        except httpx.TimeoutException as exc:
            raise HAClientError(
                f"Timeout publishing temperature to '{entity_id}'",
                status_code=504,
            ) from exc
        except httpx.RequestError as exc:
            raise HAClientError(f"Network error publishing temperature to '{entity_id}': {exc}") from exc

        if response.status_code not in (200, 201):
            raise HAClientError(
                f"Failed to publish state to '{entity_id}': HTTP {response.status_code} - {response.text}",
                status_code=response.status_code,
            )

        logger.debug("Successfully published %s = %d°C to Home Assistant", entity_id, int_value)
        return True
