import numpy as np

from utils import clean_text, fit_image_to_canvas, render_text_to_image

LANGUAGE_MARKERS = {
    "English": [" and ", " the ", " of ", " to ", " is ", " for ", " with "],
    "French": [" le ", " la ", " les ", " de ", " des ", " une ", " est ", " pour "],
    "German": [" der ", " die ", " und ", " das ", " ist ", " mit ", " eine "],
    "Italian": [" il ", " la ", " che ", " per ", " una ", " dei ", " gli ", " sono "],
    "Spanish": [" el ", " la ", " de ", " que ", " los ", " una ", " para ", " con "],
}


def char_error_rate(pred: str, target: str) -> float:
    a = clean_text(pred)
    b = clean_text(target)
    if not b:
        return 0.0 if not a else 1.0
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, start=1):
            old = dp[j]
            cost = 0 if ca == cb else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = old
    return dp[-1] / max(len(b), 1)


def repetition_ratio(text: str, n: int = 3) -> float:
    tokens = clean_text(text).split()
    if len(tokens) < n:
        return 0.0
    grams = [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return 1.0 - (len(set(grams)) / max(len(grams), 1))


def length_ratio(pred: str, target: str) -> float:
    return len(clean_text(pred)) / max(len(clean_text(target)), 1)


def pixel_f1(pred: str, target: str, max_width: int, font_size: int, padding: int) -> float:
    pred_img = render_text_to_image(pred, max_width, font_size, padding)
    true_img = render_text_to_image(target, max_width, font_size, padding)
    canvas_h = max(pred_img.size[1], true_img.size[1], max_width)
    pred_img = fit_image_to_canvas(pred_img, canvas_h, anchor="top").convert("L")
    true_img = fit_image_to_canvas(true_img, canvas_h, anchor="top").convert("L")
    p = np.array(pred_img) > 127
    t = np.array(true_img) > 127
    tp = float((p & t).sum())
    fp = float((p & ~t).sum())
    fn = float((~p & t).sum())
    return (2.0 * tp + 1e-6) / (2.0 * tp + fp + fn + 1e-6)


def prefix_cer(pred: str, target: str, n: int = 16) -> float:
    return char_error_rate(clean_text(pred)[:n], clean_text(target)[:n])


def first_token(text: str) -> str:
    tokens = clean_text(text).split()
    return tokens[0].lower() if tokens else ""


def language_marker_counts(text: str) -> dict:
    cleaned = f" {clean_text(text).lower()} "
    return {language: sum(cleaned.count(marker) for marker in markers) for language, markers in LANGUAGE_MARKERS.items()}


def language_consistency_score(text: str, expected_language: str) -> float:
    counts = language_marker_counts(text)
    total = sum(counts.values())
    if total == 0 or expected_language not in counts:
        return 0.0
    return float(counts[expected_language] / total)


def summarize_prediction(pred: str, target: str, max_width: int, font_size: int, padding: int, expected_language: str) -> dict:
    pred_clean = clean_text(pred)
    target_clean = clean_text(target)
    lang_match = pred_clean[:1].isalpha() and target_clean[:1].isalpha()
    marker_counts = language_marker_counts(pred_clean)
    best_marker_language = max(marker_counts, key=marker_counts.get) if marker_counts else expected_language
    first = first_token(pred_clean)
    return {
        "cer": char_error_rate(pred_clean, target_clean),
        "prefix_cer": prefix_cer(pred_clean, target_clean),
        "pixel_f1": pixel_f1(pred_clean, target_clean, max_width, font_size, padding),
        "repeat": repetition_ratio(pred_clean),
        "len_ratio": length_ratio(pred_clean, target_clean),
        "pred_char_len": len(pred_clean),
        "target_char_len": len(target_clean),
        "alpha_start_match": float(lang_match),
        "first_token": first,
        "lang_consistency": language_consistency_score(pred_clean, expected_language),
        "best_marker_language": best_marker_language,
        "marker_english": marker_counts["English"],
        "marker_french": marker_counts["French"],
        "marker_german": marker_counts["German"],
        "marker_italian": marker_counts["Italian"],
        "marker_spanish": marker_counts["Spanish"],
    }
