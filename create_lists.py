import os

def create_font_list():
    output = []
    exclude = {"NISC18030.ttf"}
    font_dir = '/System/Library/Fonts/Supplemental'
    for root, dirs, files in os.walk(font_dir):
        for f in files:
            if f not in exclude:
                output.append(os.path.join(root,f))
    return output

def create_character_list():
    """Return ~530 characters for OCR class labels.

    ASCII printable (U+0020-U+007E) come first to preserve backward
    compatibility with existing class IDs 4-98.  Remaining blocks cover
    Latin-1 Supplement, Latin Extended-A, Greek, math operators, arrows,
    letterlike symbols, and typographic punctuation commonly found in
    LaTeX documents.
    """
    chars = []

    # ASCII printable (95 chars) — MUST be first for backward compat
    chars.extend(chr(x) for x in range(0x0020, 0x007F))

    # Latin-1 Supplement (U+00A0-U+00FF, 96 chars)
    chars.extend(chr(x) for x in range(0x00A0, 0x0100))

    # Latin Extended-A (U+0100-U+017F, 128 chars)
    chars.extend(chr(x) for x in range(0x0100, 0x0180))

    # Greek uppercase + lowercase (U+0391-U+03A9, U+03B1-U+03C9)
    chars.extend(chr(x) for x in range(0x0391, 0x03AA))  # Alpha-Omega uppercase
    chars.extend(chr(x) for x in range(0x03B1, 0x03CA))  # alpha-omega lowercase

    # Mathematical operators (selected, ~50 chars)
    _math_ops = [
        0x2200,  # FOR ALL
        0x2202,  # PARTIAL DIFFERENTIAL
        0x2203,  # THERE EXISTS
        0x2205,  # EMPTY SET
        0x2207,  # NABLA
        0x2208,  # ELEMENT OF
        0x2209,  # NOT AN ELEMENT OF
        0x220B,  # CONTAINS AS MEMBER
        0x220F,  # N-ARY PRODUCT
        0x2211,  # N-ARY SUMMATION
        0x2212,  # MINUS SIGN
        0x2215,  # DIVISION SLASH
        0x2217,  # ASTERISK OPERATOR
        0x221A,  # SQUARE ROOT
        0x221D,  # PROPORTIONAL TO
        0x221E,  # INFINITY
        0x2220,  # ANGLE
        0x2227,  # LOGICAL AND
        0x2228,  # LOGICAL OR
        0x2229,  # INTERSECTION
        0x222A,  # UNION
        0x222B,  # INTEGRAL
        0x222C,  # DOUBLE INTEGRAL
        0x222D,  # TRIPLE INTEGRAL
        0x2234,  # THEREFORE
        0x2235,  # BECAUSE
        0x223C,  # TILDE OPERATOR
        0x2248,  # ALMOST EQUAL TO
        0x2260,  # NOT EQUAL TO
        0x2261,  # IDENTICAL TO
        0x2264,  # LESS-THAN OR EQUAL TO
        0x2265,  # GREATER-THAN OR EQUAL TO
        0x226A,  # MUCH LESS-THAN
        0x226B,  # MUCH GREATER-THAN
        0x2282,  # SUBSET OF
        0x2283,  # SUPERSET OF
        0x2286,  # SUBSET OF OR EQUAL TO
        0x2287,  # SUPERSET OF OR EQUAL TO
        0x228E,  # MULTISET UNION
        0x2291,  # SQUARE IMAGE OF OR EQUAL TO
        0x2292,  # SQUARE ORIGINAL OF OR EQUAL TO
        0x2295,  # CIRCLED PLUS
        0x2297,  # CIRCLED TIMES
        0x22A2,  # RIGHT TACK (proves)
        0x22A3,  # LEFT TACK
        0x22A5,  # UP TACK (perpendicular)
        0x22C5,  # DOT OPERATOR
        0x22C6,  # STAR OPERATOR
        0x22EE,  # VERTICAL ELLIPSIS
        0x22EF,  # MIDLINE HORIZONTAL ELLIPSIS
    ]
    chars.extend(chr(x) for x in _math_ops)

    # Arrows (selected, ~12 chars)
    _arrows = [
        0x2190,  # LEFTWARDS ARROW
        0x2191,  # UPWARDS ARROW
        0x2192,  # RIGHTWARDS ARROW
        0x2193,  # DOWNWARDS ARROW
        0x2194,  # LEFT RIGHT ARROW
        0x2195,  # UP DOWN ARROW
        0x21A6,  # RIGHTWARDS ARROW FROM BAR (mapsto)
        0x21D0,  # LEFTWARDS DOUBLE ARROW
        0x21D1,  # UPWARDS DOUBLE ARROW
        0x21D2,  # RIGHTWARDS DOUBLE ARROW
        0x21D3,  # DOWNWARDS DOUBLE ARROW
        0x21D4,  # LEFT RIGHT DOUBLE ARROW
    ]
    chars.extend(chr(x) for x in _arrows)

    # Letterlike symbols (selected, ~15 chars)
    _letterlike = [
        0x210F,  # PLANCK CONSTANT OVER TWO PI (hbar)
        0x2111,  # SCRIPT CAPITAL I (imaginary part)
        0x2113,  # SCRIPT SMALL L (ell)
        0x2115,  # DOUBLE-STRUCK CAPITAL N
        0x2119,  # DOUBLE-STRUCK CAPITAL P
        0x211A,  # DOUBLE-STRUCK CAPITAL Q
        0x211D,  # DOUBLE-STRUCK CAPITAL R
        0x2124,  # DOUBLE-STRUCK CAPITAL Z
        0x212C,  # SCRIPT CAPITAL B
        0x2130,  # SCRIPT CAPITAL E
        0x2131,  # SCRIPT CAPITAL F
        0x210B,  # SCRIPT CAPITAL H
        0x2110,  # SCRIPT CAPITAL I
        0x2112,  # SCRIPT CAPITAL L
        0x2133,  # SCRIPT CAPITAL M
    ]
    chars.extend(chr(x) for x in _letterlike)

    # Multiplication / division signs
    chars.append(chr(0x00D7))  # MULTIPLICATION SIGN (already in Latin-1 block above)
    chars.append(chr(0x00F7))  # DIVISION SIGN (already in Latin-1 block above)

    # Typographic punctuation
    _typographic = [
        0x2026,  # HORIZONTAL ELLIPSIS
        0x2013,  # EN DASH
        0x2014,  # EM DASH
        0x2018,  # LEFT SINGLE QUOTATION MARK
        0x2019,  # RIGHT SINGLE QUOTATION MARK
        0x201C,  # LEFT DOUBLE QUOTATION MARK
        0x201D,  # RIGHT DOUBLE QUOTATION MARK
    ]
    chars.extend(chr(x) for x in _typographic)

    # Prime marks
    chars.append(chr(0x2032))  # PRIME
    chars.append(chr(0x2033))  # DOUBLE PRIME

    # Deduplicate while preserving order (ASCII first)
    seen = set()
    deduped = []
    for ch in chars:
        if ch not in seen:
            seen.add(ch)
            deduped.append(ch)

    return deduped
