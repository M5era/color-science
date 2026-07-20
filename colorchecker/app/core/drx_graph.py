"""PowerGrade (.drx) node-graph surgery: add / duplicate / reorder /
relabel nodes — the generic protobuf re-serializer that template
patching (app/core/drx.py) could not do.

Format (reverse-engineered from Marc's templates, 2026-07-20; field
numbers are protobuf tags inside a decompressed body):

    body
      1  GRADE message
           1   (selected node id — left untouched)
           2,3,4,11,12  misc (left untouched)
           7   repeated NODE:
                 1  node id            (varint)
                 2  serial badge       (the "1","2",... shown in Resolve)
                 4,5  x,y in the node editor
                 6  LABEL              (plain string!)
                 7  ?  8  node kind: 44 = corrector, 90 = layer mixer
                 9  builtin corrector params (raw, kept verbatim)
                 10 OFX/DCTL payload — contains the DCTL path and the
                    sliderFloatParamN fixed-width doubles
                 12 timestamp-ish (kept verbatim)
           8   repeated EDGE: {1: from id, 3: to id, [4: dest input
                 port], 5: 64, 6: 64, 7: unique link id}
           9   ENTRY: {1: connector id, 2: 64, repeated 3:
                 {1: link id, 2: 64, 3: connector id, 4: first node}}
                 (Marc's full template has TWO connections — the source
                 feeds two branches into a layer mixer)
           10  EXIT: same shape, single connection from the last node
      other top-level fields kept verbatim

Safety discipline (same as the slider patcher): GradeGraph must
re-serialize byte-identically before any edit — tested against every
template in templates/. Resolve writes minimal varints, so a faithful
re-encode of the parsed tree reproduces the input exactly.

Generated grades are PURE SERIAL chains (Marc: no layer mixer needed
in an exported look): `serial_rebuild` rewires entry -> head prep ->
managed nodes in chain order -> display tail, drops mixers, and
reduces the entry to a single connection.
"""

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

_SLIDER_RE = re.compile(rb"sliderFloatParam(\d+)\x12\x09\x11")
_DCTL_RE = re.compile(rb"______DCTL______/[ -~]+?\.dctl")

NODE_KIND_CORRECTOR = 44
NODE_KIND_MIXER = 90


# ------------------------------------------------------ protobuf core

def _read_varint(buf: bytes, i: int) -> tuple[int, int]:
    val, shift = 0, 0
    while True:
        b = buf[i]
        val |= (b & 0x7F) << shift
        i += 1
        if not b & 0x80:
            return val, i
        shift += 7


def _write_varint(val: int) -> bytes:
    out = bytearray()
    while True:
        b = val & 0x7F
        val >>= 7
        if val:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def parse_fields(buf: bytes) -> list[tuple[int, int, object]]:
    """Flat parse: [(field number, wire type, value)]. Values: int for
    varint, bytes for everything else (fixed64/32 kept as raw bytes)."""
    out, i, n = [], 0, len(buf)
    while i < n:
        tag, i = _read_varint(buf, i)
        fnum, wt = tag >> 3, tag & 7
        if wt == 0:
            v, i = _read_varint(buf, i)
            out.append((fnum, wt, v))
        elif wt == 1:
            out.append((fnum, wt, buf[i:i + 8]))
            i += 8
        elif wt == 2:
            ln, i = _read_varint(buf, i)
            out.append((fnum, wt, buf[i:i + ln]))
            i += ln
        elif wt == 5:
            out.append((fnum, wt, buf[i:i + 4]))
            i += 4
        else:
            raise ValueError(f"unsupported wire type {wt} at offset {i}")
    return out


def serialize_fields(fields_: list[tuple[int, int, object]]) -> bytes:
    out = bytearray()
    for fnum, wt, v in fields_:
        out += _write_varint(fnum << 3 | wt)
        if wt == 0:
            out += _write_varint(v)
        elif wt == 2:
            out += _write_varint(len(v))
            out += v
        else:  # 1 / 5: raw fixed bytes
            out += v
    return bytes(out)


def _msg_get(fields_, fnum, default=None):
    for f, _, v in fields_:
        if f == fnum:
            return v
    return default


def _msg_set(fields_, fnum, wt, value):
    """Replace the first occurrence (append if absent, after the last
    field with a smaller number to keep writer-style ordering)."""
    for k, (f, w, _) in enumerate(fields_):
        if f == fnum:
            fields_[k] = (fnum, wt, value)
            return
    at = len(fields_)
    for k, (f, _, _) in enumerate(fields_):
        if f > fnum:
            at = k
            break
    fields_.insert(at, (fnum, wt, value))


# ------------------------------------------------------ graph objects

@dataclass
class GraphNode:
    fields: list                     # the node message, parsed 1 level
    _graph: "GradeGraph" = None

    @property
    def node_id(self) -> int:
        return _msg_get(self.fields, 1)

    @property
    def badge(self) -> int:
        return _msg_get(self.fields, 2, 0)

    @badge.setter
    def badge(self, v: int) -> None:
        _msg_set(self.fields, 2, 0, int(v))

    @property
    def kind(self) -> int:
        return _msg_get(self.fields, 8, NODE_KIND_CORRECTOR)

    @property
    def is_mixer(self) -> bool:
        return self.kind == NODE_KIND_MIXER

    @property
    def label(self) -> str:
        v = _msg_get(self.fields, 6, b"")
        return v.decode("utf-8", "replace") if isinstance(v, bytes) else ""

    @label.setter
    def label(self, text: str) -> None:
        _msg_set(self.fields, 6, 2, text.encode("utf-8"))

    @property
    def xy(self) -> tuple[int, int]:
        return _msg_get(self.fields, 4, 0), _msg_get(self.fields, 5, 0)

    def move(self, x: int, y: int) -> None:
        _msg_set(self.fields, 4, 0, int(x))
        _msg_set(self.fields, 5, 0, int(y))

    # ---- OFX/DCTL payload (field 10) ----

    def _ofx(self) -> bytes | None:
        v = _msg_get(self.fields, 10)
        return v if isinstance(v, bytes) else None

    @property
    def dctl_path(self) -> str | None:
        blob = self._ofx()
        if blob:
            m = _DCTL_RE.search(blob)
            if m:
                return m.group().decode()
        return None

    @property
    def dctl_name(self) -> str | None:
        p = self.dctl_path
        return Path(p).stem if p else None

    def sliders(self) -> dict[int, float]:
        blob = self._ofx() or b""
        return {int(m.group(1)):
                struct.unpack("<d", blob[m.end():m.end() + 8])[0]
                for m in _SLIDER_RE.finditer(blob)}

    def set_slider(self, index: int, value: float) -> None:
        blob = self._ofx()
        if blob is None:
            raise KeyError(f"node {self.node_id} has no OFX payload")
        for m in _SLIDER_RE.finditer(blob):
            if int(m.group(1)) == index:
                off = m.end()
                new = blob[:off] + struct.pack("<d", float(value)) \
                    + blob[off + 8:]
                _msg_set(self.fields, 10, 2, new)
                return
        raise KeyError(
            f"node {self.node_id} ({self.dctl_name}) has no "
            f"sliderFloatParam{index}"
        )


@dataclass
class GradeGraph:
    """One body's grade, parsed for surgery. Everything not understood
    is preserved verbatim; re-serialization is byte-identical until an
    edit is made (tested against all templates)."""

    top: list = field(default_factory=list)     # body top-level fields
    grade_index: int = -1                       # index of the grade in top
    pre: list = field(default_factory=list)     # grade fields before nodes
    nodes: list = field(default_factory=list)   # GraphNode
    edges: list = field(default_factory=list)   # parsed edge field lists
    entry: list = None                          # parsed connector msg
    exit: list = None
    post: list = field(default_factory=list)    # grade fields after nodes

    @classmethod
    def parse(cls, payload: bytes) -> "GradeGraph | None":
        g = cls()
        g.top = parse_fields(payload)
        for idx, (fnum, wt, v) in enumerate(g.top):
            if fnum == 1 and wt == 2:
                sub = parse_fields(v)
                if any(f == 7 for f, _, _ in sub):
                    g.grade_index = idx
                    break
        if g.grade_index < 0:
            return None
        seen_node = False
        for fnum, wt, v in parse_fields(g.top[g.grade_index][2]):
            if fnum == 7 and wt == 2:
                g.nodes.append(GraphNode(parse_fields(v), g))
                seen_node = True
            elif fnum == 8 and wt == 2:
                g.edges.append(parse_fields(v))
            elif fnum == 9 and wt == 2 and seen_node:
                g.entry = parse_fields(v)
            elif fnum == 10 and wt == 2 and seen_node:
                g.exit = parse_fields(v)
            elif not seen_node:
                g.pre.append((fnum, wt, v))
            else:
                g.post.append((fnum, wt, v))
        return g

    # ------------------------------------------------------ queries

    def node(self, node_id: int) -> GraphNode:
        for n in self.nodes:
            if n.node_id == node_id:
                return n
        raise KeyError(f"no node {node_id}")

    def entry_targets(self) -> list[int]:
        """Node ids the graph input feeds (repeated connection subs)."""
        return [_msg_get(parse_fields(v), 4)
                for f, w, v in (self.entry or []) if f == 3]

    def exit_source(self) -> int | None:
        for f, w, v in (self.exit or []):
            if f == 3:
                return _msg_get(parse_fields(v), 4)
        return None

    def successors(self, node_id: int) -> list[tuple[int, int]]:
        """[(to_id, dest input port)] for edges leaving node_id."""
        out = []
        for e in self.edges:
            if _msg_get(e, 1) == node_id:
                out.append((_msg_get(e, 3), _msg_get(e, 4, 0)))
        return out

    def main_line(self) -> list[int]:
        """Serial walk from the FIRST entry connection to the exit,
        stepping through mixers via any input."""
        targets = self.entry_targets()
        chain, cur, seen = [], (targets[0] if targets else None), set()
        while cur is not None and cur not in seen:
            seen.add(cur)
            chain.append(cur)
            nxt = self.successors(cur)
            cur = nxt[0][0] if nxt else None
        return chain

    def _max_ids(self) -> tuple[int, int]:
        max_node = max((n.node_id for n in self.nodes), default=0)
        link_ids = [_msg_get(e, 7, 0) for e in self.edges]
        for conn in (self.entry, self.exit):
            for f, w, v in (conn or []):
                if f == 3:
                    link_ids.append(_msg_get(parse_fields(v), 1, 0))
        return max_node, max(link_ids, default=0)

    # ------------------------------------------------------ surgery

    def duplicate_node(self, node_id: int) -> GraphNode:
        """Copy a node (unwired — serial_rebuild wires it). New unique
        id, same params/label, nudged position."""
        src = self.node(node_id)
        max_node, _ = self._max_ids()
        clone = GraphNode(list(src.fields), self)
        # deep-copy bytes values are immutable; the list copy suffices,
        # but the id/pos fields must not alias the source list
        clone.fields = [(f, w, v) for f, w, v in src.fields]
        _msg_set(clone.fields, 1, 0, max_node + 1)
        x, y = src.xy
        clone.move(x + 40, y + 60)
        self.nodes.append(clone)
        return clone

    def serial_rebuild(self, order: list[int]) -> None:
        """Rewire the whole grade as ONE serial chain: entry ->
        order[0] -> ... -> order[-1] -> exit. Everything not in
        `order` (layer mixers, orphaned utility nodes) is DELETED —
        generated grades are pure serial, nothing dangles. The entry
        keeps a single connection. Badges renumber 1..N in chain
        order; node editor x positions follow the chain."""
        if not order:
            raise ValueError("serial_rebuild needs at least one node")
        ids = {n.node_id for n in self.nodes}
        missing = set(order) - ids
        if missing:
            raise KeyError(f"order references unknown nodes {missing}")
        if len(set(order)) != len(order):
            raise ValueError("duplicate node id in order")
        self.nodes = [n for n in self.nodes if n.node_id in set(order)]
        self.nodes.sort(key=lambda n: order.index(n.node_id))

        _, max_link = self._max_ids()
        link = max_link

        def next_link():
            nonlocal link
            link += 1
            return link

        # fresh edges along the chain (template edge shape: 5/6 = 64)
        self.edges = []
        for a, b in zip(order, order[1:]):
            self.edges.append([(1, 0, a), (3, 0, b),
                               (5, 0, 64), (6, 0, 64),
                               (7, 0, next_link())])

        # entry: single connection -> first node (keep connector ids)
        def _rebuild_conn(conn, node_id):
            base = [(f, w, v) for f, w, v in (conn or []) if f != 3]
            sub = None
            for f, w, v in (conn or []):
                if f == 3:
                    sub = parse_fields(v)
                    break
            if sub is None:
                sub = [(1, 0, 0), (2, 0, 64), (3, 0, 0), (4, 0, 0)]
            _msg_set(sub, 1, 0, next_link())
            _msg_set(sub, 4, 0, node_id)
            base.append((3, 2, serialize_fields(sub)))
            return base

        self.entry = _rebuild_conn(self.entry, order[0])
        self.exit = _rebuild_conn(self.exit, order[-1])

        for i, nid in enumerate(order):
            n = self.node(nid)
            n.badge = i + 1
            n.move(40 + (i % 8) * 200, 60 + (i // 8) * 220)

    # ------------------------------------------------------ output

    def serialize(self) -> bytes:
        grade = list(self.pre)
        for n in self.nodes:
            grade.append((7, 2, serialize_fields(n.fields)))
        for e in self.edges:
            grade.append((8, 2, serialize_fields(e)))
        if self.entry is not None:
            grade.append((9, 2, serialize_fields(self.entry)))
        if self.exit is not None:
            grade.append((10, 2, serialize_fields(self.exit)))
        grade.extend(self.post)
        top = list(self.top)
        top[self.grade_index] = (1, 2, serialize_fields(grade))
        return serialize_fields(top)


# --------------------------------------------------- chain assembly

@dataclass
class ChainReport:
    node_id: int
    dctl_name: str
    action: str          # "fitted" | "duplicated" | "identity"
    label: str = ""


def rebuild_as_chain(graph: GradeGraph,
                     want: list[tuple[str, list, str]],
                     stage_dctl_names: set[str],
                     identity_lookup: dict[str, list],
                     ) -> list[ChainReport]:
    """Materialize a fitted chain in the grade:

    - assign each wanted stage (dctl name, params, label) a node of
      its type, DUPLICATING when the template has too few instances;
    - keep head prep nodes (non-stage nodes before the first stage on
      the main line, e.g. a Curve node) and the display tail (e.g.
      OpenDRT + 3DCube) in place;
    - reset leftover stage-type nodes to identity (kept after the
      fitted block for hand-tweaking);
    - drop layer mixers and off-chain utility nodes (Marc: generated
      grades are pure serial);
    - rewire everything serially and renumber the badges.

    Returns per-node reports for the CLI.
    """
    main = graph.main_line()
    is_stage = {n.node_id: (n.dctl_name in stage_dctl_names)
                for n in graph.nodes}
    stage_positions = [nid for nid in main if is_stage.get(nid)]
    if stage_positions:
        first = main.index(stage_positions[0])
        last = main.index(stage_positions[-1])
        head = [nid for nid in main[:first]
                if not graph.node(nid).is_mixer]
        tail = [nid for nid in main[last + 1:]
                if not graph.node(nid).is_mixer]
    else:
        head, tail = [], [nid for nid in main
                          if not graph.node(nid).is_mixer]

    # assignment pool: stage-type nodes in main-line order first, then
    # off-main stage nodes (e.g. a LiftGammaGain parked on a branch)
    pool: dict[str, list[int]] = {}
    ordered_stage_nodes = [nid for nid in main if is_stage.get(nid)] + \
        [n.node_id for n in graph.nodes
         if is_stage.get(n.node_id) and n.node_id not in main]
    for nid in ordered_stage_nodes:
        pool.setdefault(graph.node(nid).dctl_name, []).append(nid)

    reports: list[ChainReport] = []
    fitted_ids: list[int] = []
    for dctl_name, params, label in want:
        avail = pool.get(dctl_name, [])
        if avail:
            nid = avail.pop(0)
            action = "fitted"
        else:
            template_node = next(
                (n for n in graph.nodes if n.dctl_name == dctl_name), None)
            if template_node is None:
                reports.append(ChainReport(-1, dctl_name, "missing", label))
                continue
            nid = graph.duplicate_node(template_node.node_id).node_id
            action = "duplicated"
        node = graph.node(nid)
        for i, value in enumerate(params):
            if i in node.sliders():
                node.set_slider(i, float(value))
        if label:
            node.label = label
        fitted_ids.append(nid)
        reports.append(ChainReport(nid, dctl_name, action, label))

    # leftovers: unassigned stage-type nodes -> identity, after the
    # fitted block
    leftovers = [nid for nids in pool.values() for nid in nids]
    leftovers.sort(key=lambda nid: main.index(nid) if nid in main
                   else len(main))
    for nid in leftovers:
        node = graph.node(nid)
        ident = identity_lookup.get(node.dctl_name)
        if ident is not None:
            for i, value in enumerate(ident):
                if i in node.sliders():
                    node.set_slider(i, float(value))
        reports.append(ChainReport(nid, node.dctl_name, "identity",
                                   node.label))

    graph.serial_rebuild(head + fitted_ids + leftovers + tail)
    return reports


# --------------------------------------------------- template-level

def graph_bodies(template) -> dict[int, GradeGraph]:
    """Parse every body of a DrxTemplate that contains a node graph.
    Returns {body index: GradeGraph}."""
    out = {}
    for bi, (_, payload) in enumerate(template.bodies):
        g = GradeGraph.parse(bytes(payload))
        if g is not None and g.nodes:
            out[bi] = g
    return out


def write_graph(template, body_index: int, graph: GradeGraph) -> None:
    """Replace a body's payload with the re-serialized graph (the
    template's own write() then recompresses)."""
    prefix, _ = template.bodies[body_index]
    template.bodies[body_index] = (prefix, bytearray(graph.serialize()))
