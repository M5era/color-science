"""Minimal generic protobuf codec — enough to parse a .drx body into a
field tree, edit/duplicate subtrees, and re-serialize with CORRECT
length prefixes (the thing the fixed-width slider patcher can't do).

No .proto schema: every length-delimited field is kept as raw bytes and
only parsed deeper on demand. Re-serialization is byte-exact for
canonically-encoded input (which Resolve writes), so a parse -> emit
round-trip reproduces the file exactly — the correctness gate before we
start cloning nodes.
"""

from __future__ import annotations


def read_varint(buf: bytes, i: int) -> tuple[int, int]:
    val = shift = 0
    while True:
        b = buf[i]
        val |= (b & 0x7F) << shift
        i += 1
        if not b & 0x80:
            return val, i
        shift += 7


def write_varint(val: int) -> bytes:
    out = bytearray()
    while True:
        b = val & 0x7F
        val >>= 7
        if val:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


class Field:
    """One protobuf field: (number, wire_type, value). `value` is an int
    for varint/fixed wire types, or bytes for length-delimited (wire 2)."""

    __slots__ = ("number", "wire", "value")

    def __init__(self, number: int, wire: int, value):
        self.number = number
        self.wire = wire
        self.value = value

    def as_message(self) -> "Message":
        """Parse a length-delimited field's bytes as a sub-message."""
        return Message.parse(self.value)

    def __repr__(self):
        v = self.value
        if isinstance(v, (bytes, bytearray)):
            v = f"<{len(v)} bytes>"
        return f"Field({self.number}, wire={self.wire}, {v})"


class Message:
    """An ordered list of Fields. Preserves field order for byte-exact
    re-emission."""

    def __init__(self, fields: list[Field] | None = None):
        self.fields = fields if fields is not None else []

    @classmethod
    def parse(cls, buf: bytes) -> "Message":
        fields, i, n = [], 0, len(buf)
        while i < n:
            tag, i = read_varint(buf, i)
            number, wire = tag >> 3, tag & 0x7
            if wire == 0:            # varint
                value, i = read_varint(buf, i)
            elif wire == 1:          # 64-bit
                value, i = buf[i:i + 8], i + 8
            elif wire == 2:          # length-delimited
                ln, i = read_varint(buf, i)
                value, i = buf[i:i + ln], i + ln
            elif wire == 5:          # 32-bit
                value, i = buf[i:i + 4], i + 4
            else:
                raise ValueError(f"unsupported wire type {wire} at {i}")
            fields.append(Field(number, wire, value))
        return cls(fields)

    def serialize(self) -> bytes:
        out = bytearray()
        for f in self.fields:
            out += write_varint((f.number << 3) | f.wire)
            if f.wire == 0:
                out += write_varint(f.value)
            elif f.wire in (1, 5):
                out += f.value
            elif f.wire == 2:
                body = f.value
                if isinstance(body, Message):
                    body = body.serialize()
                out += write_varint(len(body))
                out += body
            else:
                raise ValueError(f"unsupported wire type {f.wire}")
        return bytes(out)

    def find(self, number: int) -> list[Field]:
        return [f for f in self.fields if f.number == number]
