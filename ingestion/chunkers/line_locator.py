"""Finds the line range a chunk covers in the file it was cut from."""

from bisect import bisect_right


class LineLocator:
    def __init__(self, text):
        self.text = text
        self._line_starts = [0] + [index + 1
                                   for index, char in enumerate(text)
                                   if char == "\n"]
        # The file as its non-blank lines, stripped, with the line number
        # each one carries. A piece is matched against this rather than
        # against the raw text: the splitters re-indent what they return,
        # and blank lines come and go.
        self._numbers = []
        self._lines = []
        for number, line in enumerate(text.splitlines(), start=1):
            if line.strip():
                self._numbers.append(number)
                self._lines.append(line.strip())
        # Where the same line occurs, so a search does not walk the file.
        self._occurrences = {}
        for index, line in enumerate(self._lines):
            self._occurrences.setdefault(line, []).append(index)
        self._cursor = 0

    # 1-based line holding `offset`.
    def line_of(self, offset):
        return bisect_right(self._line_starts, offset)

    # Line range (start, end) of `piece` inside the text, searching forward
    # from the previous match. Returns None when the piece cannot be found --
    # the caller decides what to fall back on.
    #
    # The whole run of lines is what identifies a piece. Anchoring on its
    # first line alone puts a chunk that opens on `/**`, `}` or `return` on
    # the first such line after the cursor rather than on its own: a fifth of
    # the kdk chunks used to point somewhere else in the right file.
    def locate(self, piece):
        needle = [line.strip() for line in piece.splitlines() if line.strip()]
        if not needle:
            return None
        found = _search(self._lines, self._occurrences, needle, self._cursor)
        if found is None:
            return None
        index, length = found
        # Only the start moves the cursor: the next piece may begin before
        # this one ends when the splitter overlaps them.
        self._cursor = index + 1
        return self._numbers[index], self._numbers[index + length - 1]

    # Line range of an explicit offset span.
    def span(self, start, end):
        return self.line_of(start), self.line_of(max(start, end - 1))

    # Restart the forward search at `offset`, when the caller already knows
    # which region the next pieces come from.
    def seek(self, offset):
        line = self.line_of(offset)
        self._cursor = bisect_right(self._numbers, line - 1)

    # The whole file, for a chunk that cannot be traced to a narrower span.
    def whole_file(self):
        return 1, max(1, self.line_of(len(self.text) - 1))


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------


# Find `needle` -- a run of stripped lines -- in `lines`, at or after
# `cursor`, and return (index, matched length).
#
# A splitter can cut a piece short, so the run is shortened from the end
# until it matches; the longest match wins, and a single line is accepted
# only when nothing longer does.
def _search(lines, occurrences, needle, cursor):
    for length in range(len(needle), 0, -1):
        index = _find_run(lines, occurrences, needle[:length], cursor)
        if index is not None:
            return index, length
    return None


# The first index at or after `cursor` where `run` matches, falling back to a
# search from the top so a piece the splitter reordered is still placed.
def _find_run(lines, occurrences, run, cursor):
    candidates = occurrences.get(run[0], ())
    ahead = [index for index in candidates if index >= cursor]
    for index in ahead + [index for index in candidates if index < cursor]:
        if lines[index:index + len(run)] == run:
            return index
    return None
