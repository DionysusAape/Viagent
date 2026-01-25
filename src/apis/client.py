"""response for HTTP requests"""

from typing import Any, Dict, Optional

import requests


class VideoEvidenceClient:
    def __init__(self, base_url: str, timeout: float = 60.0):
        self.base_url = base_url
        self.timeout = timeout

    def ping(self) -> Dict[str, Any]:
        r = requests.get(f"{self.base_url}/ping", timeout=min(self.timeout, 30))
        r.raise_for_status()
        return r.json()

    def list_videos(self, label: str = "all", generator: Optional[str] = None, limit: int = 500) -> Dict[str, Any]:
        params = {"label": label, "limit": limit}
        if generator:
            params["generator"] = generator
        r = requests.get(f"{self.base_url}/videos", params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def get_evidence(self, video_id: str, include_data_urls: bool = False) -> Dict[str, Any]:
        params = {"include_data_urls": str(include_data_urls).lower()}
        r = requests.get(f"{self.base_url}/videos/{video_id}/evidence", params=params, timeout=max(self.timeout, 120))
        r.raise_for_status()
        return r.json()
