"""Locate a chunk back in the file it was cut from.

A chunk is only useful if a developer can open the file at the right place,
so every chunk carries the line range it covers. The splitters return text,
not offsets, and they strip whitespace as they go -- so the text has to be
found again in the source. Doing that naively is where line numbers go
wrong: searching from `previous_offset + len(previous_piece)` overshoots the
start of the next chunk whenever the splitter overlaps them, the search
fails, and every following chunk inherits a drifting position.

Locator searches forward from just after the previous match instead, which
holds whether or not the pieces overlap.
"""

from bisect import bisect_right


class Locator:
    def __init__(self, text):
        self.text = text
        self._line_starts = [0] + [index + 1
                                   for index, char in enumerate(text)
                                   if char == "\n"]
        self._cursor = 0

    # 1-based line holding `offset`.
    def line_of(self, offset):
        return bisect_right(self._line_starts, offset)

    # Line range (start, end) of `piece` inside the text, searching forward
    # from the previous match. Returns None when the piece cannot be found --
    # the caller decides what to fall back on.
    def locate(self, piece):
        head = _first_line(piece)
        tail = _last_line(piece)
        if not head:
            return None
        start = self._find(head)
        if start is None:
            return None
        # The end is looked for backwards from where the piece must end: a
        # last line as common as "}" would otherwise match the first closing
        # brace after the start and cut the range short.
        window_end = start + len(piece) + len(tail)
        end = self.text.rfind(tail, start, window_end)
        end = start + len(piece) if end < 0 else end + len(tail)
        # Only the start moves the cursor: the next piece may begin before
        # this one ends when the splitter overlaps them.
        self._cursor = start + 1
        return self.line_of(start), self.line_of(max(start, end - 1))

    # Line range of an explicit offset span.
    def span(self, start, end):
        return self.line_of(start), self.line_of(max(start, end - 1))

    # Restart the forward search at `offset`, when the caller already knows
    # which region the next pieces come from.
    def seek(self, offset):
        self._cursor = offset

    # The whole file, for a chunk that cannot be traced to a narrower span.
    def whole_file(self):
        return 1, max(1, self.line_of(len(self.text) - 1))

    # ------------------------------------------------------------------
    # UTILS
    # ------------------------------------------------------------------

    # Forward search, falling back to a search from the top so a piece the
    # splitter reordered is still placed somewhere real.
    def _find(self, needle):
        offset = self.text.find(needle, self._cursor)
        if offset < 0:
            offset = self.text.find(needle)
        return None if offset < 0 else offset


# ---------------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------------


# First non-blank line of a piece; that is what anchors its start.
def _first_line(piece):
    for line in piece.splitlines():
        if line.strip():
            return line
    return ""


# Last non-blank line of a piece; that is what anchors its end.
def _last_line(piece):
    for line in reversed(piece.splitlines()):
        if line.strip():
            return line
    return ""
