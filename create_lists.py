"""Character list and font mapping utilities for OCR training.

Provides two character sets:
- ``create_character_list()`` — legacy ~530 Latin/Greek/math chars (backward compat)
- ``create_unicode_character_list()`` — full Unicode coverage from installed fonts
"""

import os
import unicodedata

# ---------------------------------------------------------------------------
# Legacy functions
# ---------------------------------------------------------------------------

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
        0x2200, 0x2202, 0x2203, 0x2205, 0x2207, 0x2208, 0x2209, 0x220B,
        0x220F, 0x2211, 0x2212, 0x2215, 0x2217, 0x221A, 0x221D, 0x221E,
        0x2220, 0x2227, 0x2228, 0x2229, 0x222A, 0x222B, 0x222C, 0x222D,
        0x2234, 0x2235, 0x223C, 0x2248, 0x2260, 0x2261, 0x2264, 0x2265,
        0x226A, 0x226B, 0x2282, 0x2283, 0x2286, 0x2287, 0x228E, 0x2291,
        0x2292, 0x2295, 0x2297, 0x22A2, 0x22A3, 0x22A5, 0x22C5, 0x22C6,
        0x22EE, 0x22EF,
    ]
    chars.extend(chr(x) for x in _math_ops)

    # Arrows (selected, ~12 chars)
    _arrows = [
        0x2190, 0x2191, 0x2192, 0x2193, 0x2194, 0x2195,
        0x21A6, 0x21D0, 0x21D1, 0x21D2, 0x21D3, 0x21D4,
    ]
    chars.extend(chr(x) for x in _arrows)

    # Letterlike symbols (selected, ~15 chars)
    _letterlike = [
        0x210F, 0x2111, 0x2113, 0x2115, 0x2119, 0x211A, 0x211D, 0x2124,
        0x212C, 0x2130, 0x2131, 0x210B, 0x2110, 0x2112, 0x2133,
    ]
    chars.extend(chr(x) for x in _letterlike)

    # Multiplication / division signs
    chars.append(chr(0x00D7))
    chars.append(chr(0x00F7))

    # Typographic punctuation
    _typographic = [0x2026, 0x2013, 0x2014, 0x2018, 0x2019, 0x201C, 0x201D]
    chars.extend(chr(x) for x in _typographic)

    # Prime marks
    chars.append(chr(0x2032))
    chars.append(chr(0x2033))

    # Deduplicate while preserving order (ASCII first)
    seen = set()
    deduped = []
    for ch in chars:
        if ch not in seen:
            seen.add(ch)
            deduped.append(ch)

    return deduped

# Alias for backward compat
create_character_list_legacy = create_character_list


# ---------------------------------------------------------------------------
# Unicode block utilities
# ---------------------------------------------------------------------------

# Unicode block ranges (name → (start, end)) — covers all blocks through Unicode 15.1
# We define them programmatically from unicodedata when possible, but the block
# boundaries themselves must be hard-coded since Python's unicodedata doesn't
# expose block ranges directly.

def _get_unicode_block(cp):
    """Return the Unicode block name for codepoint *cp*.

    Uses a lookup against the standard block boundaries.
    """
    for name, (start, end) in _UNICODE_BLOCKS:
        if start <= cp <= end:
            return name
    return "Unknown"


def _build_block_table():
    """Build Unicode block table from the Blocks.txt-style ranges.

    Returns list of (name, (start, end)) sorted by start.
    """
    # Major blocks — sufficient for classification grouping.
    # We don't need every single block, just enough to group chars sensibly.
    blocks = [
        ("Basic Latin", 0x0000, 0x007F),
        ("Latin-1 Supplement", 0x0080, 0x00FF),
        ("Latin Extended-A", 0x0100, 0x017F),
        ("Latin Extended-B", 0x0180, 0x024F),
        ("IPA Extensions", 0x0250, 0x02AF),
        ("Spacing Modifier Letters", 0x02B0, 0x02FF),
        ("Combining Diacritical Marks", 0x0300, 0x036F),
        ("Greek and Coptic", 0x0370, 0x03FF),
        ("Cyrillic", 0x0400, 0x04FF),
        ("Cyrillic Supplement", 0x0500, 0x052F),
        ("Armenian", 0x0530, 0x058F),
        ("Hebrew", 0x0590, 0x05FF),
        ("Arabic", 0x0600, 0x06FF),
        ("Syriac", 0x0700, 0x074F),
        ("Arabic Supplement", 0x0750, 0x077F),
        ("Thaana", 0x0780, 0x07BF),
        ("NKo", 0x07C0, 0x07FF),
        ("Samaritan", 0x0800, 0x083F),
        ("Mandaic", 0x0840, 0x085F),
        ("Syriac Supplement", 0x0860, 0x086F),
        ("Arabic Extended-B", 0x0870, 0x089F),
        ("Arabic Extended-A", 0x08A0, 0x08FF),
        ("Devanagari", 0x0900, 0x097F),
        ("Bengali", 0x0980, 0x09FF),
        ("Gurmukhi", 0x0A00, 0x0A7F),
        ("Gujarati", 0x0A80, 0x0AFF),
        ("Oriya", 0x0B00, 0x0B7F),
        ("Tamil", 0x0B80, 0x0BFF),
        ("Telugu", 0x0C00, 0x0C7F),
        ("Kannada", 0x0C80, 0x0CFF),
        ("Malayalam", 0x0D00, 0x0D7F),
        ("Sinhala", 0x0D80, 0x0DFF),
        ("Thai", 0x0E00, 0x0E7F),
        ("Lao", 0x0E80, 0x0EFF),
        ("Tibetan", 0x0F00, 0x0FFF),
        ("Myanmar", 0x1000, 0x109F),
        ("Georgian", 0x10A0, 0x10FF),
        ("Hangul Jamo", 0x1100, 0x11FF),
        ("Ethiopic", 0x1200, 0x137F),
        ("Ethiopic Supplement", 0x1380, 0x139F),
        ("Cherokee", 0x13A0, 0x13FF),
        ("Unified Canadian Aboriginal Syllabics", 0x1400, 0x167F),
        ("Ogham", 0x1680, 0x169F),
        ("Runic", 0x16A0, 0x16FF),
        ("Tagalog", 0x1700, 0x171F),
        ("Hanunoo", 0x1720, 0x173F),
        ("Buhid", 0x1740, 0x175F),
        ("Tagbanwa", 0x1760, 0x177F),
        ("Khmer", 0x1780, 0x17FF),
        ("Mongolian", 0x1800, 0x18AF),
        ("Unified Canadian Aboriginal Syllabics Extended", 0x18B0, 0x18FF),
        ("Limbu", 0x1900, 0x194F),
        ("Tai Le", 0x1950, 0x197F),
        ("New Tai Lue", 0x1980, 0x19DF),
        ("Khmer Symbols", 0x19E0, 0x19FF),
        ("Buginese", 0x1A00, 0x1A1F),
        ("Tai Tham", 0x1A20, 0x1AAF),
        ("Combining Diacritical Marks Extended", 0x1AB0, 0x1AFF),
        ("Balinese", 0x1B00, 0x1B7F),
        ("Sundanese", 0x1B80, 0x1BBF),
        ("Batak", 0x1BC0, 0x1BFF),
        ("Lepcha", 0x1C00, 0x1C4F),
        ("Ol Chiki", 0x1C50, 0x1C7F),
        ("Cyrillic Extended-C", 0x1C80, 0x1C8F),
        ("Georgian Extended", 0x1C90, 0x1CBF),
        ("Sundanese Supplement", 0x1CC0, 0x1CCF),
        ("Vedic Extensions", 0x1CD0, 0x1CFF),
        ("Phonetic Extensions", 0x1D00, 0x1D7F),
        ("Phonetic Extensions Supplement", 0x1D80, 0x1DBF),
        ("Combining Diacritical Marks Supplement", 0x1DC0, 0x1DFF),
        ("Latin Extended Additional", 0x1E00, 0x1EFF),
        ("Greek Extended", 0x1F00, 0x1FFF),
        ("General Punctuation", 0x2000, 0x206F),
        ("Superscripts and Subscripts", 0x2070, 0x209F),
        ("Currency Symbols", 0x20A0, 0x20CF),
        ("Combining Diacritical Marks for Symbols", 0x20D0, 0x20FF),
        ("Letterlike Symbols", 0x2100, 0x214F),
        ("Number Forms", 0x2150, 0x218F),
        ("Arrows", 0x2190, 0x21FF),
        ("Mathematical Operators", 0x2200, 0x22FF),
        ("Miscellaneous Technical", 0x2300, 0x23FF),
        ("Control Pictures", 0x2400, 0x243F),
        ("Optical Character Recognition", 0x2440, 0x245F),
        ("Enclosed Alphanumerics", 0x2460, 0x24FF),
        ("Box Drawing", 0x2500, 0x257F),
        ("Block Elements", 0x2580, 0x259F),
        ("Geometric Shapes", 0x25A0, 0x25FF),
        ("Miscellaneous Symbols", 0x2600, 0x26FF),
        ("Dingbats", 0x2700, 0x27BF),
        ("Miscellaneous Mathematical Symbols-A", 0x27C0, 0x27EF),
        ("Supplemental Arrows-A", 0x27F0, 0x27FF),
        ("Braille Patterns", 0x2800, 0x28FF),
        ("Supplemental Arrows-B", 0x2900, 0x297F),
        ("Miscellaneous Mathematical Symbols-B", 0x2980, 0x29FF),
        ("Supplemental Mathematical Operators", 0x2A00, 0x2AFF),
        ("Miscellaneous Symbols and Arrows", 0x2B00, 0x2BFF),
        ("Glagolitic", 0x2C00, 0x2C5F),
        ("Latin Extended-C", 0x2C60, 0x2C7F),
        ("Coptic", 0x2C80, 0x2CFF),
        ("Georgian Supplement", 0x2D00, 0x2D2F),
        ("Tifinagh", 0x2D30, 0x2D7F),
        ("Ethiopic Extended", 0x2D80, 0x2DDF),
        ("Cyrillic Extended-A", 0x2DE0, 0x2DFF),
        ("Supplemental Punctuation", 0x2E00, 0x2E7F),
        ("CJK Radicals Supplement", 0x2E80, 0x2EFF),
        ("Kangxi Radicals", 0x2F00, 0x2FDF),
        ("Ideographic Description Characters", 0x2FF0, 0x2FFF),
        ("CJK Symbols and Punctuation", 0x3000, 0x303F),
        ("Hiragana", 0x3040, 0x309F),
        ("Katakana", 0x30A0, 0x30FF),
        ("Bopomofo", 0x3100, 0x312F),
        ("Hangul Compatibility Jamo", 0x3130, 0x318F),
        ("Kanbun", 0x3190, 0x319F),
        ("Bopomofo Extended", 0x31A0, 0x31BF),
        ("CJK Strokes", 0x31C0, 0x31EF),
        ("Katakana Phonetic Extensions", 0x31F0, 0x31FF),
        ("Enclosed CJK Letters and Months", 0x3200, 0x32FF),
        ("CJK Compatibility", 0x3300, 0x33FF),
        ("CJK Unified Ideographs Extension A", 0x3400, 0x4DBF),
        ("Yijing Hexagram Symbols", 0x4DC0, 0x4DFF),
        ("CJK Unified Ideographs", 0x4E00, 0x9FFF),
        ("Yi Syllables", 0xA000, 0xA48F),
        ("Yi Radicals", 0xA490, 0xA4CF),
        ("Lisu", 0xA4D0, 0xA4FF),
        ("Vai", 0xA500, 0xA63F),
        ("Cyrillic Extended-B", 0xA640, 0xA69F),
        ("Bamum", 0xA6A0, 0xA6FF),
        ("Modifier Tone Letters", 0xA700, 0xA71F),
        ("Latin Extended-D", 0xA720, 0xA7FF),
        ("Syloti Nagri", 0xA800, 0xA82F),
        ("Common Indic Number Forms", 0xA830, 0xA83F),
        ("Phags-pa", 0xA840, 0xA87F),
        ("Saurashtra", 0xA880, 0xA8DF),
        ("Devanagari Extended", 0xA8E0, 0xA8FF),
        ("Kayah Li", 0xA900, 0xA92F),
        ("Rejang", 0xA930, 0xA95F),
        ("Hangul Jamo Extended-A", 0xA960, 0xA97F),
        ("Javanese", 0xA980, 0xA9DF),
        ("Myanmar Extended-B", 0xA9E0, 0xA9FF),
        ("Cham", 0xAA00, 0xAA5F),
        ("Myanmar Extended-A", 0xAA60, 0xAA7F),
        ("Tai Viet", 0xAA80, 0xAADF),
        ("Meetei Mayek Extensions", 0xAAE0, 0xAAFF),
        ("Ethiopic Extended-A", 0xAB00, 0xAB2F),
        ("Latin Extended-E", 0xAB30, 0xAB6F),
        ("Cherokee Supplement", 0xAB70, 0xABBF),
        ("Meetei Mayek", 0xABC0, 0xABFF),
        ("Hangul Syllables", 0xAC00, 0xD7AF),
        ("Hangul Jamo Extended-B", 0xD7B0, 0xD7FF),
        ("CJK Compatibility Ideographs", 0xF900, 0xFAFF),
        ("Alphabetic Presentation Forms", 0xFB00, 0xFB4F),
        ("Arabic Presentation Forms-A", 0xFB50, 0xFDFF),
        ("Variation Selectors", 0xFE00, 0xFE0F),
        ("Vertical Forms", 0xFE10, 0xFE1F),
        ("Combining Half Marks", 0xFE20, 0xFE2F),
        ("CJK Compatibility Forms", 0xFE30, 0xFE4F),
        ("Small Form Variants", 0xFE50, 0xFE6F),
        ("Arabic Presentation Forms-B", 0xFE70, 0xFEFF),
        ("Halfwidth and Fullwidth Forms", 0xFF00, 0xFFEF),
        ("Specials", 0xFFF0, 0xFFFF),
        # Supplementary Multilingual Plane (SMP)
        ("Linear B Syllabary", 0x10000, 0x1007F),
        ("Linear B Ideograms", 0x10080, 0x100FF),
        ("Aegean Numbers", 0x10100, 0x1013F),
        ("Ancient Greek Numbers", 0x10140, 0x1018F),
        ("Ancient Symbols", 0x10190, 0x101CF),
        ("Phaistos Disc", 0x101D0, 0x101FF),
        ("Lycian", 0x10280, 0x1029F),
        ("Carian", 0x102A0, 0x102DF),
        ("Coptic Epact Numbers", 0x102E0, 0x102FF),
        ("Old Italic", 0x10300, 0x1032F),
        ("Gothic", 0x10330, 0x1034F),
        ("Old Permic", 0x10350, 0x1037F),
        ("Ugaritic", 0x10380, 0x1039F),
        ("Old Persian", 0x103A0, 0x103DF),
        ("Deseret", 0x10400, 0x1044F),
        ("Shavian", 0x10450, 0x1047F),
        ("Osmanya", 0x10480, 0x104AF),
        ("Osage", 0x104B0, 0x104FF),
        ("Elbasan", 0x10500, 0x1052F),
        ("Caucasian Albanian", 0x10530, 0x1056F),
        ("Vithkuqi", 0x10570, 0x105BF),
        ("Latin Extended-F", 0x10780, 0x107BF),
        ("Cyrillic Extended-D", 0x10800, 0x1083F),  # reused range
        ("Old Turkic", 0x10C00, 0x10C4F),
        ("Old Hungarian", 0x10C80, 0x10CFF),
        ("Hanifi Rohingya", 0x10D00, 0x10D3F),
        ("Arabic Extended-C", 0x10EC0, 0x10EFF),
        ("Old Sogdian", 0x10F00, 0x10F2F),
        ("Sogdian", 0x10F30, 0x10F6F),
        ("Old Uyghur", 0x10F70, 0x10FAF),
        ("Chorasmian", 0x10FB0, 0x10FDF),
        ("Elymaic", 0x10FE0, 0x10FFF),
        ("Brahmi", 0x11000, 0x1107F),
        ("Kaithi", 0x11080, 0x110CF),
        ("Sora Sompeng", 0x110D0, 0x110FF),
        ("Chakma", 0x11100, 0x1114F),
        ("Mahajani", 0x11150, 0x1117F),
        ("Sharada", 0x11180, 0x111DF),
        ("Sinhala Archaic Numbers", 0x111E0, 0x111FF),
        ("Khojki", 0x11200, 0x1124F),
        ("Multani", 0x11280, 0x112AF),
        ("Khudawadi", 0x112B0, 0x112FF),
        ("Grantha", 0x11300, 0x1137F),
        ("Newa", 0x11400, 0x1147F),
        ("Tirhuta", 0x11480, 0x114DF),
        ("Siddham", 0x11580, 0x115FF),
        ("Modi", 0x11600, 0x1165F),
        ("Mongolian Supplement", 0x11660, 0x1167F),
        ("Takri", 0x11680, 0x116CF),
        ("Ahom", 0x11700, 0x1174F),
        ("Dogra", 0x11800, 0x1184F),
        ("Warang Citi", 0x118A0, 0x118FF),
        ("Dives Akuru", 0x11900, 0x1195F),
        ("Nandinagari", 0x119A0, 0x119FF),
        ("Zanabazar Square", 0x11A00, 0x11A4F),
        ("Soyombo", 0x11A50, 0x11AAF),
        ("UCAS Extended-A", 0x11AB0, 0x11ABF),
        ("Pau Cin Hau", 0x11AC0, 0x11AFF),
        ("Bhaiksuki", 0x11C00, 0x11C6F),
        ("Marchen", 0x11C70, 0x11CBF),
        ("Masaram Gondi", 0x11D00, 0x11D5F),
        ("Gunjala Gondi", 0x11D60, 0x11DAF),
        ("Makasar", 0x11EE0, 0x11EFF),
        ("Kawi", 0x11F00, 0x11F5F),
        ("Tamil Supplement", 0x11FC0, 0x11FFF),
        ("Cuneiform", 0x12000, 0x123FF),
        ("Cuneiform Numbers and Punctuation", 0x12400, 0x1247F),
        ("Early Dynastic Cuneiform", 0x12480, 0x1254F),
        ("Cypro-Minoan", 0x12F90, 0x12FFF),
        ("Egyptian Hieroglyphs", 0x13000, 0x1342F),
        ("Anatolian Hieroglyphs", 0x14400, 0x1467F),
        ("Bamum Supplement", 0x16800, 0x16A3F),
        ("Mro", 0x16A40, 0x16A6F),
        ("Tangsa", 0x16A70, 0x16ACF),
        ("Bassa Vah", 0x16AD0, 0x16AFF),
        ("Pahawh Hmong", 0x16B00, 0x16B8F),
        ("Medefaidrin", 0x16E40, 0x16E9F),
        ("Miao", 0x16F00, 0x16F9F),
        ("Ideographic Symbols and Punctuation", 0x16FE0, 0x16FFF),
        ("Tangut", 0x17000, 0x187FF),
        ("Tangut Components", 0x18800, 0x18AFF),
        ("Khitan Small Script", 0x18B00, 0x18CFF),
        ("Tangut Supplement", 0x18D00, 0x18D7F),
        ("Kana Extended-B", 0x1AFF0, 0x1AFFF),
        ("Kana Supplement", 0x1B000, 0x1B0FF),
        ("Kana Extended-A", 0x1B100, 0x1B12F),
        ("Small Kana Extension", 0x1B130, 0x1B16F),
        ("Nushu", 0x1B170, 0x1B2FF),
        ("Duployan", 0x1BC00, 0x1BC9F),
        ("Shorthand Format Controls", 0x1BCA0, 0x1BCAF),
        ("Znamenny Musical Notation", 0x1CF00, 0x1CFCF),
        ("Byzantine Musical Symbols", 0x1D000, 0x1D0FF),
        ("Musical Symbols", 0x1D100, 0x1D1FF),
        ("Ancient Greek Musical Notation", 0x1D200, 0x1D24F),
        ("Kaktovik Numerals", 0x1D2C0, 0x1D2DF),
        ("Mayan Numerals", 0x1D2E0, 0x1D2FF),
        ("Tai Xuan Jing Symbols", 0x1D300, 0x1D35F),
        ("Counting Rod Numerals", 0x1D360, 0x1D37F),
        ("Mathematical Alphanumeric Symbols", 0x1D400, 0x1D7FF),
        ("Sutton SignWriting", 0x1D800, 0x1DAAF),
        ("Latin Extended-G", 0x1DF00, 0x1DFFF),
        ("Glagolitic Supplement", 0x1E000, 0x1E02F),
        ("Cyrillic Extended-D2", 0x1E030, 0x1E08F),
        ("Nyiakeng Puachue Hmong", 0x1E100, 0x1E14F),
        ("Toto", 0x1E290, 0x1E2BF),
        ("Wancho", 0x1E2C0, 0x1E2FF),
        ("Nag Mundari", 0x1E4D0, 0x1E4FF),
        ("Ethiopic Extended-B", 0x1E7E0, 0x1E7FF),
        ("Mende Kikakui", 0x1E800, 0x1E8DF),
        ("Adlam", 0x1E900, 0x1E95F),
        ("Indic Siyaq Numbers", 0x1EC70, 0x1ECBF),
        ("Ottoman Siyaq Numbers", 0x1ED00, 0x1ED4F),
        ("Arabic Mathematical Alphabetic Symbols", 0x1EE00, 0x1EEFF),
        ("Mahjong Tiles", 0x1F000, 0x1F02F),
        ("Domino Tiles", 0x1F030, 0x1F09F),
        ("Playing Cards", 0x1F0A0, 0x1F0FF),
        ("Enclosed Alphanumeric Supplement", 0x1F100, 0x1F1FF),
        ("Enclosed Ideographic Supplement", 0x1F200, 0x1F2FF),
        ("Miscellaneous Symbols and Pictographs", 0x1F300, 0x1F5FF),
        ("Emoticons", 0x1F600, 0x1F64F),
        ("Ornamental Dingbats", 0x1F650, 0x1F67F),
        ("Transport and Map Symbols", 0x1F680, 0x1F6FF),
        ("Alchemical Symbols", 0x1F700, 0x1F77F),
        ("Geometric Shapes Extended", 0x1F780, 0x1F7FF),
        ("Supplemental Arrows-C", 0x1F800, 0x1F8FF),
        ("Supplemental Symbols and Pictographs", 0x1F900, 0x1F9FF),
        ("Chess Symbols", 0x1FA00, 0x1FA6F),
        ("Symbols and Pictographs Extended-A", 0x1FA70, 0x1FAFF),
        ("Symbols for Legacy Computing", 0x1FB00, 0x1FBFF),
        # Supplementary Ideographic Plane (SIP)
        ("CJK Unified Ideographs Extension B", 0x20000, 0x2A6DF),
        ("CJK Unified Ideographs Extension C", 0x2A700, 0x2B73F),
        ("CJK Unified Ideographs Extension D", 0x2B740, 0x2B81F),
        ("CJK Unified Ideographs Extension E", 0x2B820, 0x2CEAF),
        ("CJK Unified Ideographs Extension F", 0x2CEB0, 0x2EBEF),
        ("CJK Compatibility Ideographs Supplement", 0x2F800, 0x2FA1F),
        # Tertiary Ideographic Plane (TIP)
        ("CJK Unified Ideographs Extension G", 0x30000, 0x3134F),
        ("CJK Unified Ideographs Extension H", 0x31350, 0x323AF),
    ]
    return [(name, (start, end)) for name, start, end in blocks]


_UNICODE_BLOCKS = _build_block_table()

# Build name→index mapping
BLOCK_NAMES = [name for name, _ in _UNICODE_BLOCKS]
BLOCK_NAME_TO_IDX = {name: i for i, name in enumerate(BLOCK_NAMES)}
NUM_BLOCKS = len(BLOCK_NAMES)


# ---------------------------------------------------------------------------
# Full Unicode character list from installed fonts
# ---------------------------------------------------------------------------

def _scan_font_codepoints(font_path, face_index=0):
    """Extract all renderable codepoints from a font file using fonttools cmap."""
    try:
        from fontTools.ttLib import TTFont
        font = TTFont(font_path, fontNumber=face_index)
        cmap = font.getBestCmap()
        if cmap is None:
            font.close()
            return set()
        cps = set(cmap.keys())
        font.close()
        return cps
    except Exception:
        return set()


def create_unicode_character_list(font_faces=None):
    """Build full Unicode character list from all installed fonts.

    Args:
        font_faces: list of (path, face_index) tuples from discover_fonts_all().
                    If None, scans default font directories.

    Returns:
        (chars, char_to_fonts, block_char_counts) where:
        - chars: sorted list of unique characters renderable by ≥1 font
        - char_to_fonts: dict mapping char → list of (path, face_index)
        - block_char_counts: dict mapping block_name → number of chars in that block
    """
    if font_faces is None:
        from generate_training_data import discover_fonts_all
        font_faces = discover_fonts_all(latin_only=False)

    # Scan all fonts for codepoints
    char_to_fonts = {}  # char → list of (path, face_index)
    for path, face_idx in font_faces:
        cps = _scan_font_codepoints(path, face_idx)
        for cp in cps:
            # Skip control characters, surrogates, private use, non-characters
            if cp < 0x20:
                continue
            if 0x7F <= cp <= 0x9F:  # C1 controls
                continue
            if 0xD800 <= cp <= 0xDFFF:  # surrogates
                continue
            if 0xE000 <= cp <= 0xF8FF:  # private use area
                continue
            if 0xF0000 <= cp <= 0x10FFFF:  # supplementary private use
                continue
            cat = unicodedata.category(chr(cp))
            if cat.startswith('C'):  # control/format/unassigned
                continue

            ch = chr(cp)
            if ch not in char_to_fonts:
                char_to_fonts[ch] = []
            char_to_fonts[ch].append((path, face_idx))

    # Sort by codepoint, but ASCII printable first for backward compat
    all_chars = sorted(char_to_fonts.keys(), key=lambda c: ord(c))

    # Ensure ASCII printable (0x20-0x7E) come first in exact order
    ascii_chars = [chr(x) for x in range(0x20, 0x7F) if chr(x) in char_to_fonts]
    non_ascii = [c for c in all_chars if ord(c) >= 0x7F]
    chars = ascii_chars + non_ascii

    # Count chars per block
    block_char_counts = {}
    for ch in chars:
        block = _get_unicode_block(ord(ch))
        block_char_counts[block] = block_char_counts.get(block, 0) + 1

    return chars, char_to_fonts, block_char_counts


def get_block_for_char(ch):
    """Return the Unicode block name for a character."""
    return _get_unicode_block(ord(ch))


def get_block_index(ch):
    """Return the block index (0-based) for a character."""
    name = _get_unicode_block(ord(ch))
    return BLOCK_NAME_TO_IDX.get(name, -1)


def build_block_char_map(chars):
    """Build mapping from block_index → list of (char, within_block_index).

    Returns:
        block_to_chars: dict mapping block_idx → sorted list of chars in that block
        char_to_block_local: dict mapping char → (block_idx, local_idx_within_block)
    """
    from collections import defaultdict
    block_to_chars = defaultdict(list)
    for ch in chars:
        block_idx = get_block_index(ch)
        block_to_chars[block_idx].append(ch)

    # Sort within each block by codepoint
    for block_idx in block_to_chars:
        block_to_chars[block_idx].sort(key=lambda c: ord(c))

    # Build reverse map: char → (block_idx, local_idx)
    char_to_block_local = {}
    for block_idx, block_chars in block_to_chars.items():
        for local_idx, ch in enumerate(block_chars):
            char_to_block_local[ch] = (block_idx, local_idx)

    return dict(block_to_chars), char_to_block_local


# ---------------------------------------------------------------------------
# CLI: --stats mode
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Character list utilities")
    parser.add_argument("--stats", action="store_true",
                        help="Print Unicode character coverage statistics")
    parser.add_argument("--legacy", action="store_true",
                        help="Print legacy character list info")
    args = parser.parse_args()

    if args.legacy:
        chars = create_character_list()
        print(f"Legacy character list: {len(chars)} characters")
        print(f"  ASCII: {sum(1 for c in chars if 0x20 <= ord(c) <= 0x7E)}")
        print(f"  Non-ASCII: {sum(1 for c in chars if ord(c) > 0x7E)}")
        return

    if args.stats:
        print("Scanning all installed fonts for Unicode coverage...")
        from generate_training_data import discover_fonts_all
        font_faces = discover_fonts_all(latin_only=False)
        print(f"Found {len(font_faces)} font faces")

        chars, char_to_fonts, block_counts = create_unicode_character_list(font_faces)
        print(f"\nTotal renderable characters: {len(chars)}")

        # Block breakdown
        print(f"\nBlock coverage ({len(block_counts)} blocks with chars):")
        print(f"{'Block':<50} {'Chars':>7}  {'Fonts>50%':>9}")
        print("-" * 70)
        for block_name in BLOCK_NAMES:
            count = block_counts.get(block_name, 0)
            if count == 0:
                continue
            # Count fonts that cover >50% of chars in this block
            block_chars = [c for c in chars if _get_unicode_block(ord(c)) == block_name]
            font_coverage = {}  # (path, idx) → count
            for ch in block_chars:
                for font_key in char_to_fonts.get(ch, []):
                    font_coverage[font_key] = font_coverage.get(font_key, 0) + 1
            threshold = count * 0.5
            fonts_over_50 = sum(1 for c in font_coverage.values() if c >= threshold)
            print(f"  {block_name:<48} {count:>7}  {fonts_over_50:>9}")

        # Build block map for classifier
        block_to_chars, char_to_block_local = build_block_char_map(chars)
        active_blocks = len(block_to_chars)
        max_block_size = max(len(v) for v in block_to_chars.values())
        print(f"\nBlocks with ≥1 char: {active_blocks}")
        print(f"Largest block: {max_block_size} chars")
        # Top 5 blocks by size
        top_blocks = sorted(block_to_chars.items(),
                           key=lambda x: len(x[1]), reverse=True)[:10]
        print("\nTop 10 blocks by size:")
        for block_idx, block_chars in top_blocks:
            if block_idx < 0 or block_idx >= len(BLOCK_NAMES):
                name = "Unknown/Unmapped"
            else:
                name = BLOCK_NAMES[block_idx]
            print(f"  {name}: {len(block_chars)} chars")
        return

    # Default: print legacy list
    chars = create_character_list()
    print(f"Legacy: {len(chars)} chars")


if __name__ == "__main__":
    main()
