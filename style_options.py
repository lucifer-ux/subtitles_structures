FONT_OPTIONS = {
    "bold_sans": "Montserrat ExtraBold",
    "rounded": "Poppins Bold",
}

ACCENT_OPTIONS = {
    "yellow": "#FFD300",
    "mint": "#7CF5C4",
    "pink": "#FF3B84",
}

POSITION_OPTIONS = {
    "standard": {"y_percent": 62, "x_margin_percent": 12},
    "low": {"y_percent": 72, "x_margin_percent": 15},
    "compact": {"y_percent": 70, "x_margin_percent": 18},
}

FONT_SIZE_OPTIONS = {
    "standard": 5.5,
    "subtle": 4.5,
    "compact": 3.5,
}

ANIMATION_OPTIONS = {
    "word_highlight": {"type": "color_shift", "transition_ms": 0},
    "paint_reveal": {"type": "clip_wipe", "transition_ms": 80},
    "fade_word": {"type": "opacity_fade", "transition_ms": 100, "inactive_opacity": 0.5},
    "line_replace": {"type": "static_line", "transition_ms": 0},
}


def build_style(font_key, accent_key, position_key, font_size_key="subtle", animation_key="word_highlight"):
    if font_key not in FONT_OPTIONS:
        raise KeyError("Invalid font option")
    if accent_key not in ACCENT_OPTIONS:
        raise KeyError("Invalid accent option")
    if position_key not in POSITION_OPTIONS:
        raise KeyError("Invalid position option")
    if font_size_key not in FONT_SIZE_OPTIONS:
        raise KeyError("Invalid font size option")
    if animation_key not in ANIMATION_OPTIONS:
        raise KeyError("Invalid animation option")
    return {
        "font": FONT_OPTIONS[font_key],
        "fill": "#FFFFFF",
        "outline": {"color": "#000000", "width": 2},
        "accent": ACCENT_OPTIONS[accent_key],
        "words_visible": 2,
        "position": POSITION_OPTIONS[position_key],
        "font_size_percent_of_height": FONT_SIZE_OPTIONS[font_size_key],
        "animation": ANIMATION_OPTIONS[animation_key],
    }


def public_options():
    return {
        "fonts": sorted(FONT_OPTIONS),
        "accents": sorted(ACCENT_OPTIONS),
        "positions": sorted(POSITION_OPTIONS),
        "font_sizes": sorted(FONT_SIZE_OPTIONS),
        "animations": sorted(ANIMATION_OPTIONS),
    }
