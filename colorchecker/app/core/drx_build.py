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


def _clone_node(node: Field, new_id: int, f12: int,
                label: str | None = None) -> Field:
    """Deep-copy a field-7 node, stamping a fresh id / field12 / uuid,
    and optionally its on-screen node label (field 6, a wire-2 string
    sitting between field 5 and field 7)."""
    m = copy.deepcopy(node.as_message())
    old_id = m.find(1)[0].value
    m.find(1)[0].value = new_id
    if m.find(12):
        m.find(12)[0].value = f12
    if label is not None:
        _set_label(m, label)
    raw = m.serialize()
    # retarget the uuid suffix ...edbd_<oldid> -> ...edbd_<newid>. Same
    # digit count (both 3) so the enclosing length is unchanged.
    raw = raw.replace(f"_{old_id}".encode(), f"_{new_id}".encode())
    return Field(7, 2, raw)


def _set_label(m: Message, label: str) -> None:
    """Set a node's display label (field 6). Replace it if present, else
    insert it just after field 5 to match Resolve's field order."""
    enc = label.encode()
    existing = m.find(6)
    if existing:
        existing[0].value = enc
        return
    lbl = Field(6, 2, enc)
    after5 = 0
    for idx, f in enumerate(m.fields):
        if f.number <= 5:
            after5 = idx + 1
    m.fields.insert(after5, lbl)


def build_grade(template_path: str | Path, dctl_names: list[str],
                out_path: str | Path, drt_name: str = "OpenDRT",
                labels: list[str] | None = None) -> None:
    """Write a .drx whose node stack is exactly `dctl_names` (in order)
    followed by the DRT node, cloned from `template_path`'s library.
    Slider values are NOT set here — open the result with DrxTemplate and
    patch, as the exporter already does (k-th node of each type).

    `labels`, if given, must be parallel to `dctl_names + [drt_name]` and
    becomes each node's on-screen Resolve label; None keeps the cloned
    node's own label (usually blank)."""
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
        if labels is not None and len(labels) != len(wanted):
            raise ValueError(
                f"labels ({len(labels)}) must match nodes ({len(wanted)})")

        new_nodes, ids = [], []
        for k, name in enumerate(wanted):
            nid = _ID_BASE + k
            lbl = labels[k] if labels is not None else None
            new_nodes.append(
                _clone_node(library[name], nid, _F12_BASE + k, lbl))
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

        # the container's input/output pointers reference the FIRST and
        # LAST node by id — field 1 = output id, field 9 -> first id,
        # field 10 -> last id. Retarget them or Resolve dereferences
        # dropped nodes and crashes.
        if contm.find(1):
            contm.find(1)[0].value = ids[-1]
        _set_endpoint(contm, 9, ids[0])
        _set_endpoint(contm, 10, ids[-1])
        cont.value = contm.serialize()
        tpl.bodies[bi] = (prefix, bytearray(body.serialize()))
        tpl.nodes = tpl._scan_nodes()
        tpl.write(out_path)
        return
    raise ValueError("no node-stack container found in template")


def _set_endpoint(contm: Message, field_num: int, node_id: int) -> None:
    """Retarget the node-id reference buried in a container endpoint
    pointer (field 9/10 = {..., 3: {..., 4: <node id>}})."""
    fs = contm.find(field_num)
    if not fs:
        return
    outer = fs[0].as_message()
    if outer.find(3):
        inner = outer.find(3)[0].as_message()
        if inner.find(4):
            inner.find(4)[0].value = node_id
            outer.find(3)[0].value = inner.serialize()
    fs[0].value = outer.serialize()


def _DCTL_of(node: Field) -> str:
    m = re.search(rb"______DCTL______/[ -~]+?\.dctl", node.value)
    return m.group().decode() if m else ""
