"""Build a PowerGrade (.drx) node stack FROM SCRATCH.

The fixed-width slider patcher (drx.py) can only fill nodes that already
exist in a template. This module instead CLONES nodes out of a template
that carries one of every DCTL ("kitchen-sink"), so it can emit exactly
the fitted chain — any node types, any counts, any order — then hand the
result to DrxTemplate for the (tested) slider patch.

A .drx body's node stack lives in field 1 -> field 7 (the nodes) and
field 8 (the series connections). Each node is uniquely identified by
field 1 (id), field 12 (a unique number) and its UUID suffix
"...edbd_<id>"; connections carry from=field 1, to=field 3, and a unique
id=field 7. Cloning a node = deep-copy its field-7 message, stamp a fresh
id/uuid/field12; wiring = one field-8 record per adjacent pair. Verified
in Resolve on a single-clone file.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

from app.core.drx import DrxTemplate
from app.core.protobuf import Field, Message

_ID_BASE = 300          # fresh node ids (3 digits, keeps the uuid suffix length stable)
_F12_BASE = 900_000_000_000
_CONN_BASE = 5000
_UUID_RE = re.compile(rb"([0-9a-f]{12})_(\d+)")   # ...edbd_<id>


def _clone_node(node: Field, new_id: int, f12: int) -> Field:
    """Deep-copy a field-7 node, stamping a fresh id / field12 / uuid."""
    m = copy.deepcopy(node.as_message())
    old_id = m.find(1)[0].value
    m.find(1)[0].value = new_id
    if m.find(12):
        m.find(12)[0].value = f12
    raw = m.serialize()
    # retarget the uuid suffix ...edbd_<oldid> -> ...edbd_<newid>. Same
    # digit count (both 3) so the enclosing length is unchanged.
    raw = raw.replace(f"_{old_id}".encode(), f"_{new_id}".encode())
    return Field(7, 2, raw)


def build_grade(template_path: str | Path, dctl_names: list[str],
                out_path: str | Path, drt_name: str = "OpenDRT") -> None:
    """Write a .drx whose node stack is exactly `dctl_names` (in order)
    followed by the DRT node, cloned from `template_path`'s library.
    Slider values are NOT set here — open the result with DrxTemplate and
    patch, as the exporter already does (k-th node of each type)."""
    tpl = DrxTemplate(template_path)
    # the node stack lives in whichever body has the field-1 container
    for bi, (prefix, payload) in enumerate(tpl.bodies):
        body = Message.parse(bytes(payload))
        conts = body.find(1)
        if not conts:
            continue
        cont = conts[0]
        contm = cont.as_message()
        lib_nodes = contm.find(7)
        lib_conns = contm.find(8)
        # the node stack is the container with BOTH nodes and series
        # connections (body0's field 1 has a field 7 but no field 8)
        if not lib_nodes or not lib_conns:
            continue

        # library: one field-7 node per DCTL basename
        library = {}
        for n in lib_nodes:
            name = Path(_DCTL_of(n)).stem
            library.setdefault(name, n)
        conn_proto = lib_conns[0]

        wanted = list(dctl_names) + [drt_name]
        missing = [w for w in wanted if w not in library]
        if missing:
            raise KeyError(f"template lacks a node for: {missing}")

        new_nodes, ids = [], []
        for k, name in enumerate(wanted):
            nid = _ID_BASE + k
            new_nodes.append(_clone_node(library[name], nid, _F12_BASE + k))
            ids.append(nid)

        new_conns = []
        for k in range(len(ids) - 1):          # series: node k -> node k+1
            cm = copy.deepcopy(conn_proto.as_message())
            cm.find(1)[0].value = ids[k]
            cm.find(3)[0].value = ids[k + 1]
            if cm.find(7):
                cm.find(7)[0].value = _CONN_BASE + k
            new_conns.append(Field(8, 2, cm.serialize()))

        # rebuild the container: keep the scalar fields, swap the whole
        # node/connection run in at the first 7/8 position
        rebuilt, inserted = [], False
        for f in contm.fields:
            if f.number in (7, 8):
                if not inserted:
                    rebuilt.extend(new_nodes)
                    rebuilt.extend(new_conns)
                    inserted = True
                continue
            rebuilt.append(f)
        contm.fields = rebuilt
        cont.value = contm.serialize()
        tpl.bodies[bi] = (prefix, bytearray(body.serialize()))
        tpl.nodes = tpl._scan_nodes()
        tpl.write(out_path)
        return
    raise ValueError("no node-stack container found in template")


def _DCTL_of(node: Field) -> str:
    m = re.search(rb"______DCTL______/[ -~]+?\.dctl", node.value)
    return m.group().decode() if m else ""
