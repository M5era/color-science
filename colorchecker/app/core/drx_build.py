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
import struct
from pathlib import Path

from app.core.drx import DrxTemplate
from app.core.protobuf import Field, Message

_ID_BASE = 300          # fresh node ids (3 digits, keeps the uuid suffix length stable)
_F12_BASE = 900_000_000_000
_CONN_BASE = 5000
_UUID_RE = re.compile(rb"([0-9a-f]{12})_(\d+)")   # ...edbd_<id>

# ------------------------------------------------- node SYNTHESIS specs
#
# Node types absent from every saved template can be SYNTHESIZED: clone
# any DCTL node as a scaffold and rewrite its param map at the protobuf
# level — the map is a generic name -> typed-value KV list ("DCTLs" path
# string, sliderFloatParamN doubles, checkBoxParamN varints, comboBoxParamN
# enums), and Resolve reads entries BY NAME (Marc's own templates carry
# honored sliderFloatParam10/11 entries, so indices past 9 work). Values
# here are each DCTL's UI defaults — fitted sliders get patched afterwards
# by the normal DrxTemplate pass; a synthesized node stores a double for
# EVERY slider, so nothing is ever a "wiggle it in Resolve" template gap.
SYNTH_SPECS = {
    "FilmicContrast": {
        "path": "______DCTL______/0_MS/FilmicContrast.dctl",
        "sliders": [0.0, 1.0, 0.0, 1.015, 0.6, 6.0, 1.02, 0.8, 6.0,
                    0.0001, 0.5, 2.0, 0.0, 0.25, 2.0,
                    0.0, 0.5, 0.0, 0.0, 0.0],
        # bypass, show curve, show ramp, wide overlays, tone-mapped exp,
        # PRESERVE MID-GRAY (on), show pin range
        "checkboxes": [0, 0, 0, 0, 0, 1, 0],
        "combos": [0],          # transfer function: ARRI LogC3
    },
    "SplitTone": {
        "path": "______DCTL______/0_MS/SplitTone.dctl",
        "sliders": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        "checkboxes": [0],      # show curve
        "combos": [2],          # transfer function: Log-C3
    },
}

_PARAM_NAME_RE = re.compile(
    rb"(sliderFloat|checkBox|comboBox|intSlider)Param\d+$")


def _kv_entry(name: bytes, value_field: Field) -> Field:
    """One param-map entry: {1: name, 2: {typed value}} as a field-5."""
    entry = Message([Field(1, 2, name),
                     Field(2, 2, Message([value_field]).serialize())])
    return Field(5, 2, entry.serialize())


def _spec_entries(spec) -> list[tuple[bytes, Field]]:
    out = []
    for i, v in enumerate(spec["sliders"]):      # {2: double} (wire 1)
        out.append((f"sliderFloatParam{i}".encode(),
                    _kv_entry(f"sliderFloatParam{i}".encode(),
                              Field(2, 1, struct.pack("<d", float(v))))))
    for i, v in enumerate(spec["checkboxes"]):   # {3: varint}
        out.append((f"checkBoxParam{i}".encode(),
                    _kv_entry(f"checkBoxParam{i}".encode(),
                              Field(3, 0, int(v)))))
    for i, v in enumerate(spec["combos"]):       # {4: {1: varint}}
        out.append((f"comboBoxParam{i}".encode(),
                    _kv_entry(f"comboBoxParam{i}".encode(),
                              Field(4, 2, Message([Field(1, 0, int(v))])
                                    .serialize()))))
    return out


def _rewrite_param_block(raw: bytes, spec) -> bytes | None:
    """Recursively locate the KV param map (the message whose field-5
    entries include name == b"DCTLs") and rewrite it for `spec`: repoint
    the DCTL path, drop the scaffold's slider/checkbox/combo entries,
    insert the spec's. Entries stay ASCII-name-sorted like Resolve
    writes them. Returns the rewritten bytes, or None if `raw` doesn't
    contain the map."""
    if b"DCTLs" not in raw:
        return None
    try:
        m = Message.parse(raw)
    except (ValueError, IndexError):
        return None

    def entry_name(f):
        try:
            names = f.as_message().find(1)
            return bytes(names[0].value) if names else None
        except (ValueError, IndexError):
            return None

    is_map = any(f.number == 5 and f.wire == 2 and entry_name(f) == b"DCTLs"
                 for f in m.fields)
    if is_map:
        keep: list[tuple[bytes, Field]] = []
        for f in m.fields:
            if f.number != 5 or f.wire != 2:
                continue
            name = entry_name(f)
            if name == b"DCTLs":
                e = f.as_message()
                e.find(2)[0].value = Message(
                    [Field(5, 2, spec["path"].encode())]).serialize()
                keep.append((name, Field(5, 2, e.serialize())))
            elif name is None or _PARAM_NAME_RE.match(name):
                continue                      # re-synthesized below
            else:
                keep.append((name, f))
        keep.extend(_spec_entries(spec))
        keep.sort(key=lambda t: t[0])         # Resolve writes name-sorted
        rebuilt, inserted = [], False
        for f in m.fields:
            if f.number == 5 and f.wire == 2:
                if not inserted:
                    rebuilt.extend(kv for _, kv in keep)
                    inserted = True
                continue
            rebuilt.append(f)
        m.fields = rebuilt
        return m.serialize()

    for f in m.fields:                        # descend toward the map
        if f.wire == 2 and isinstance(f.value, (bytes, bytearray)) \
                and b"DCTLs" in f.value:
            new = _rewrite_param_block(bytes(f.value), spec)
            if new is not None:
                f.value = new
                return m.serialize()
    return None


def _synthesize_node(scaffold: Field, spec) -> Field:
    """Clone `scaffold` (any DCTL node) into a fresh node of the spec'd
    type. Id/uuid/label stamping happens later in _clone_node, exactly
    as for real library nodes."""
    new_raw = _rewrite_param_block(bytes(scaffold.value), spec)
    if new_raw is None:
        raise ValueError("scaffold node has no DCTL param map")
    return Field(7, 2, new_raw)


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
        for w in list(missing):
            if w in SYNTH_SPECS:
                scaffold = min(
                    (n for k, n in library.items() if k != drt_name),
                    key=lambda f: len(f.value))
                library[w] = _synthesize_node(scaffold, SYNTH_SPECS[w])
                missing.remove(w)
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
