import json
import pdb

from .bounding_volume import OrientedBoundingBox
from .tile import Tile

from urllib.parse import urlparse, parse_qs
import requests


def _parse(root, target_volume):
    assert "contents" not in root, "contents array not supported"

    if "children" in root:
        for child in root["children"]:
            bv = OrientedBoundingBox.from_tilespec(child["boundingVolume"])
            if target_volume.intersects(bv):
                yield from _parse(child, target_volume)
    elif "content" in root:
        yield (root["content"], root["boundingVolume"])


class TileApi:
    def __init__(self, key, working_dir, api="https://tile.googleapis.com"):
        self.key = key
        self.api = api
        self.session = None
        self.json_counter = 0
        self.working_dir = working_dir

    def get(self, target_volume, uri="/v1/3dtiles/root.json", boundingVolume=None):
        fetcher = lambda: requests.get(
            f"{self.api}{uri}",
            params={'key': self.key, 'session': self.session if uri != "/v1/3dtiles/root.json" else None},
        )

        # We got a glTF tile. Don't immediately download it, but end the recursion here.
        if uri.endswith(".glb"):
            yield Tile(uri=uri, download_thunk=fetcher, boundingVolume=boundingVolume)
            return

        response = fetcher()

        if not response.ok:
            raise RuntimeError(f"response not ok: {response.status_code}, {response.text}")

        content_type = response.headers.get("content-type")
        if content_type != "application/json":
            raise RuntimeError(f"expected JSON response but got {content_type}")

        data = response.json()
        json.dump(data, open(f"{self.working_dir}/{str(self.json_counter)}.json", "w"))
        self.json_counter += 1

        # Parse response
        for content, boundingVolume in _parse(data["root"], target_volume):
            if "uri" in content:
                uri = urlparse(content["uri"])
                # Update session ID from child URI (this is usually only returned as the URI of the
                # child of the initial root request)
                self.session = parse_qs(uri.query).get("session", [self.session])[0]
                # Recurse into child tiles
                yield from self.get(target_volume, uri.path, boundingVolume=boundingVolume if uri.path.endswith(".glb") else None)
            else:
                raise RuntimeError(f"unsupported content: {content}")
