"""Finding a chunk back in the file it was cut from.

Every line range in the index comes from here, so the failure modes matter:
overlapping pieces, a piece the splitter stripped, a repeated line, a last
line as common as a closing brace.
"""

from ingestion.chunkers.line_locator import LineLocator

TEXT = """import _ from 'lodash'

export function first () {
  return 1
}

export function second () {
  return 2
}
"""


# --- line_of ---------------------------------------------------------------

def test_the_first_character_is_on_line_one():
    assert LineLocator(TEXT).line_of(0) == 1


def test_an_offset_maps_to_the_line_holding_it():
    locator = LineLocator(TEXT)

    assert locator.line_of(TEXT.index("export function first")) == 3
    assert locator.line_of(TEXT.index("export function second")) == 7


def test_the_line_of_a_newline_is_the_line_it_ends():
    locator = LineLocator(TEXT)

    assert locator.line_of(TEXT.index("\n")) == 1


# --- locate ----------------------------------------------------------------

def test_a_piece_is_located_at_its_lines():
    piece = "export function first () {\n  return 1\n}"

    assert LineLocator(TEXT).locate(piece) == (3, 5)


def test_a_stripped_piece_is_still_located():
    # The splitters strip the pieces they return.
    piece = "  return 1\n"

    assert LineLocator(TEXT).locate(piece.strip()) == (4, 4)


def test_a_single_line_piece_reports_one_line():
    assert LineLocator(TEXT).locate("import _ from 'lodash'") == (1, 1)


def test_a_piece_that_is_not_in_the_text_is_not_located():
    assert LineLocator(TEXT).locate("export function third () {}") is None


def test_a_blank_piece_is_not_located():
    assert LineLocator(TEXT).locate("   \n\n") is None


def test_successive_pieces_advance_through_the_file():
    locator = LineLocator(TEXT)

    first = locator.locate("export function first () {\n  return 1\n}")
    second = locator.locate("export function second () {\n  return 2\n}")

    assert first == (3, 5)
    assert second == (7, 9)


def test_overlapping_pieces_are_both_located():
    # The cursor advances by one character past a match, not past the whole
    # piece: a chunk that overlaps the previous one still starts inside it.
    locator = LineLocator(TEXT)

    locator.locate("export function first () {\n  return 1\n}")
    overlapping = locator.locate("}\n\nexport function second () {")

    assert overlapping == (5, 7)


def test_a_repeated_line_resolves_to_the_next_occurrence():
    text = "value = 1\nvalue = 1\nvalue = 1\n"
    locator = LineLocator(text)

    assert locator.locate("value = 1") == (1, 1)
    assert locator.locate("value = 1") == (2, 2)
    assert locator.locate("value = 1") == (3, 3)


def test_a_common_closing_line_does_not_cut_the_range_short():
    # "}" appears many times; the end is looked for backwards from where the
    # piece must end, not forwards from its start.
    piece = "export function first () {\n  return 1\n}\n\nexport function second () {\n  return 2\n}"

    assert LineLocator(TEXT).locate(piece) == (3, 9)


def test_seek_restarts_the_search_further_down():
    locator = LineLocator(TEXT)
    locator.seek(TEXT.index("export function second"))

    # The first function is above the cursor; the fallback search from the
    # top still finds it rather than losing it.
    assert locator.locate("  return 2") == (8, 8)


# --- span and whole_file ---------------------------------------------------

def test_a_span_maps_offsets_to_lines():
    start = TEXT.index("export function first")
    end = TEXT.index("export function second")

    assert LineLocator(TEXT).span(start, end) == (3, 6)


def test_an_empty_span_reports_a_single_line():
    start = TEXT.index("export function first")

    assert LineLocator(TEXT).span(start, start) == (3, 3)


def test_the_whole_file_covers_every_line():
    assert LineLocator(TEXT).whole_file() == (1, len(TEXT.splitlines()))


def test_the_whole_file_of_an_empty_text_is_line_one():
    assert LineLocator("").whole_file() == (1, 1)


# --- a repeated opening line -----------------------------------------------

# An object literal whose members are each introduced by the same docblock
# opening, which is the shape a Leaflet plugin or a KDK mixin has.
LITERAL = "\n".join([
    "const mapProto = {",                       # 1
    "",                                         # 2
    "    /**",                                  # 3
    "     * Converts a container point.",       # 4
    "     */",                                  # 5
    "    containerPoint: function (point) {",   # 6
    "        return point",                     # 7
    "    },",                                   # 8
    "",                                         # 9
    "    /**",                                  # 10
    "     * Converts a rotated point.",         # 11
    "     */",                                  # 12
    "    rotatedPoint: function (point) {",     # 13
    "        return point.rotate(this._bearing)",  # 14
    "    },",                                   # 15
    "}",                                        # 16
])


def test_a_piece_is_placed_by_its_own_run_when_its_first_line_repeats():
    # The splitter strips the indentation, so the piece opens on a bare
    # "/**" that occurs twice in the file. Anchoring on that first line puts
    # the piece on the first docblock after the cursor -- a real function,
    # just not this one -- and the whole chunk is then attributed to the
    # wrong lines. The run of lines is what tells the two apart.
    locator = LineLocator(LITERAL)
    locator.locate("const mapProto = {")

    piece = ("/**\n     * Converts a rotated point.\n     */\n"
             "    rotatedPoint: function (point) {\n"
             "        return point.rotate(this._bearing)\n    },")

    assert locator.locate(piece) == (10, 15)


def test_the_first_of_two_identical_openings_is_still_found():
    locator = LineLocator(LITERAL)
    locator.locate("const mapProto = {")

    piece = ("/**\n     * Converts a container point.\n     */\n"
             "    containerPoint: function (point) {\n"
             "        return point\n    },")

    assert locator.locate(piece) == (3, 8)


def test_two_pieces_in_a_row_land_on_their_own_lines():
    # What a chunker actually does: locate each piece in turn, the cursor
    # carrying over. The second must not be dragged back onto the first.
    locator = LineLocator(LITERAL)
    first = locator.locate(
        "/**\n     * Converts a container point.\n     */\n"
        "    containerPoint: function (point) {\n        return point\n    },")
    second = locator.locate(
        "/**\n     * Converts a rotated point.\n     */\n"
        "    rotatedPoint: function (point) {\n"
        "        return point.rotate(this._bearing)\n    },")

    assert (first, second) == ((3, 8), (10, 15))


def test_a_piece_the_splitter_cut_short_still_lands_on_its_start():
    # A piece too long for the chunk size loses its tail; what is left has
    # to place it, and the range covers what survived.
    locator = LineLocator(LITERAL)

    piece = ("/**\n     * Converts a rotated point.\n     */\n"
             "    rotatedPoint: function (point) {\n"
             "        return point.rotate(this._bearing)\n"
             "    },\n    somethingTheSplitterInvented: true")

    assert locator.locate(piece) == (10, 15)


def test_a_piece_that_is_not_in_the_file_is_not_placed():
    assert LineLocator(LITERAL).locate("nothing like this exists") is None
