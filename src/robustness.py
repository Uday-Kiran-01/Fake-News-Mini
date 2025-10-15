# src/robustness_quick.py
import re, random

def add_spacing_noise(text: str) -> str:
    return re.sub(r"\s", lambda m: m.group(0) * 2, text)

def punct_noise(text: str) -> str:
    return text.replace(".", "..").replace(",", ",,")

def leetspeak(text: str, p: float = 0.05) -> str:
    mapping = {"a":"4","e":"3","i":"1","o":"0","s":"5","t":"7"}
    out = []
    for ch in text:
        if ch.lower() in mapping and random.random() < p:
            out.append(mapping[ch.lower()])
        else:
            out.append(ch)
    return "".join(out)
