"""PowerGrade (.drx) template patching.

A .drx is an XML wrapper holding zstd-compressed protobuf bodies (each
<Body> = one prefix byte + a zstd frame). Inside, every ResolveFX DCTL
node carries its DCTL path ("______DCTL______/0_MS/SectorSkew.dctl")
and generic OFX sliders serialized as

    "sliderFloatParamN" 12 09 11 <8-byte little-endian double>

— fixed width, so patching slider values NEVER changes the payload
length. sliderFloatParamN is the N-th DEFINE_UI_PARAMS slider of the
DCTL, and our stages' param_names are in exactly that order, so a
fitted stage maps 1:1 onto its node.

Flow (feasibility proven in Resolve on the K64 file, confirmed on
Marc's example_powergrade): clone a template .drx that already stacks
the wanted DCTL nodes, write fitted values into the doubles,
recompress, done. Nodes are matched to stages by DCTL filename.
"""

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

_BODY_RE = re.compile(r"<Body>([0-9a-fA-F]+)</Body>")
_DCTL_RE = re.compile(rb"______DCTL______/[ -~]+?\.dctl")
_SLIDER_RE = re.compile(rb"sliderFloatParam(\d+)\x12\x09\x11")


@dataclass
class DrxNode:
    dctl_path: str        # as stored in the grade
    dctl_name: str        # basename without extension, e.g. SectorSkew
    body_index: int
    sliders: dict = field(default_factory=dict)   # index -> value
    _offsets: dict = field(default_factory=dict)  # index -> byte offset


class DrxTemplate:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.text = self.path.read_text()
        import zstandard

        self._dec = zstandard
        self.bodies = []      # list of (prefix byte, bytearray payload)
        for hexdata in _BODY_RE.findall(self.text):
            blob = bytes.fromhex(hexdata)
            payload = zstandard.ZstdDecompressor().decompress(
                blob[1:], max_output_size=200_000_000
            )
            self.bodies.append((blob[0], bytearray(payload)))
        self.nodes = self._scan_nodes()

    def _scan_nodes(self) -> list[DrxNode]:
        nodes = []
        for bi, (_, payload) in enumerate(self.bodies):
            anchors = [(m.start(), m.group().decode())
                       for m in _DCTL_RE.finditer(payload)]
            for k, (pos, dctl_path) in enumerate(anchors):
                end = anchors[k + 1][0] if k + 1 < len(anchors) else len(payload)
                node = DrxNode(
                    dctl_path=dctl_path,
                    dctl_name=Path(dctl_path).stem,
                    body_index=bi,
                )
                for m in _SLIDER_RE.finditer(payload, pos, end):
                    idx = int(m.group(1))
                    off = m.end()
                    node.sliders[idx] = struct.unpack(
                        "<d", payload[off:off + 8]
                    )[0]
                    node._offsets[idx] = off
                nodes.append(node)
        return nodes

    def set_slider(self, node: DrxNode, index: int, value: float) -> None:
        if index not in node._offsets:
            raise KeyError(
                f"{node.dctl_name} has no sliderFloatParam{index} — "
                f"available: {sorted(node._offsets)}"
            )
        off = node._offsets[index]
        _, payload = self.bodies[node.body_index]
        payload[off:off + 8] = struct.pack("<d", float(value))
        node.sliders[index] = float(value)

    def write(self, out_path: str | Path) -> None:
        import zstandard

        text = self.text
        old_hexes = _BODY_RE.findall(self.text)
        compressor = zstandard.ZstdCompressor(level=19)
        for old_hex, (prefix, payload) in zip(old_hexes, self.bodies):
            blob = bytes([prefix]) + compressor.compress(bytes(payload))
            text = text.replace(old_hex, blob.hex(), 1)
        Path(out_path).write_text(text)
