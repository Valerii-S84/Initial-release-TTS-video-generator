"""
Standalone demo for voice selection with a small catalog and CLI run.
This file is independent from backend/ and safe to run directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class VoiceGender(str, Enum):
    MASCULINE = "masculine"
    FEMININE = "feminine"
    NEUTRAL = "neutral"


class VoiceAge(str, Enum):
    YOUNG = "young"
    MIDDLE_AGED = "middle_aged"
    MATURE = "mature"


class VoiceTone(str, Enum):
    WARM = "warm"
    AUTHORITATIVE = "authoritative"
    ENERGETIC = "energetic"
    CALM = "calm"
    EMOTIONAL = "emotional"
    NEUTRAL = "neutral"


@dataclass
class VoiceProfile:
    voice_id: str
    name: str
    gender: VoiceGender
    age: VoiceAge
    primary_tones: List[VoiceTone]
    languages: Optional[List[str]] = None
    description: str = ""

    def __post_init__(self) -> None:
        if self.languages is None:
            self.languages = ["en"]


VOICE_CATALOG: Dict[str, VoiceProfile] = {
    "Rachel": VoiceProfile(
        voice_id="EXAVITQu4vr4xnSDxMaL",
        name="Rachel",
        gender=VoiceGender.FEMININE,
        age=VoiceAge.MIDDLE_AGED,
        primary_tones=[VoiceTone.WARM, VoiceTone.CALM, VoiceTone.NEUTRAL],
        languages=["en"],
        description=(
            "Warm, professional voice with clear articulation. "
            "Excellent for narration and inspirational content."
        ),
    ),
    "Antoni": VoiceProfile(
        voice_id="ErXwobaYiN019PkySvjV",
        name="Antoni",
        gender=VoiceGender.MASCULINE,
        age=VoiceAge.MIDDLE_AGED,
        primary_tones=[VoiceTone.AUTHORITATIVE, VoiceTone.CALM, VoiceTone.NEUTRAL],
        languages=["en"],
        description=(
            "Deep, authoritative voice with gravitas. "
            "Ideal for serious content and professional narration."
        ),
    ),
    "Adam": VoiceProfile(
        voice_id="pNInz6obpgDQGcFmaJgB",
        name="Adam",
        gender=VoiceGender.MASCULINE,
        age=VoiceAge.YOUNG,
        primary_tones=[VoiceTone.ENERGETIC, VoiceTone.WARM],
        languages=["en"],
        description=(
            "Youthful, energetic voice with enthusiasm. "
            "Perfect for motivational and upbeat content."
        ),
    ),
    "Elli": VoiceProfile(
        voice_id="MF3mGyEYCl7XYWbV9V6O",
        name="Elli",
        gender=VoiceGender.FEMININE,
        age=VoiceAge.YOUNG,
        primary_tones=[VoiceTone.EMOTIONAL, VoiceTone.WARM],
        languages=["en"],
        description=(
            "Expressive, emotional voice with range. "
            "Great for storytelling and emotional narratives."
        ),
    ),
}


@dataclass
class VoiceSelectionCriteria:
    voice_type: Optional[str] = None
    energy_level: Optional[str] = None
    emotion: Optional[str] = None
    formality: Optional[str] = None
    tone: Optional[str] = None
    gender: Optional[str] = None
    age: Optional[str] = None
    language: str = "en"
    prefer_neutral: bool = False


class VoiceSelector:
    def __init__(self, catalog: Optional[Dict[str, VoiceProfile]] = None) -> None:
        self.catalog = catalog or VOICE_CATALOG
        self.logger = logging.getLogger(__name__)

    def _score(self, name: str, profile: VoiceProfile, c: VoiceSelectionCriteria) -> float:
        score = 0.0
        vt_map = {
            "warm": [VoiceTone.WARM, VoiceTone.EMOTIONAL],
            "authoritative": [VoiceTone.AUTHORITATIVE, VoiceTone.CALM],
            "dynamic": [VoiceTone.ENERGETIC],
            "gentle": [VoiceTone.WARM, VoiceTone.CALM],
            "neutral": [VoiceTone.NEUTRAL, VoiceTone.CALM],
        }
        if c.voice_type:
            targets = vt_map.get(c.voice_type.lower(), [])
            if any(t in profile.primary_tones for t in targets):
                score += 20.0
        if c.energy_level:
            if c.energy_level.lower() == "energetic" and VoiceTone.ENERGETIC in profile.primary_tones:
                score += 10.0
            if c.energy_level.lower() == "calm" and VoiceTone.CALM in profile.primary_tones:
                score += 10.0
        if c.emotion and VoiceTone.EMOTIONAL in profile.primary_tones:
            score += 5.0
        if c.language and profile.languages and c.language in profile.languages:
            score += 5.0
        return score

    def select(self, c: VoiceSelectionCriteria) -> VoiceProfile:
        scores = {name: self._score(name, p, c) for name, p in self.catalog.items()}
        best = max(scores, key=scores.get)
        if scores[best] > 0:
            return self.catalog[best]
        # Legacy fallback
        if c.tone:
            tone = c.tone.lower()
            if tone in ("warm", "emotional"):
                return self.catalog["Rachel"]
            if tone in ("serious", "authoritative"):
                return self.catalog["Antoni"]
            if tone in ("energetic", "dynamic"):
                return self.catalog["Adam"]
        return self.catalog["Rachel"]


class VoiceSelectorForQuoteAnalyzer:
    def __init__(self) -> None:
        self.selector = VoiceSelector()

    def select_from_dict(self, analysis_dict: Dict) -> VoiceProfile:
        c = VoiceSelectionCriteria(
            voice_type=analysis_dict.get("voice_type"),
            energy_level=analysis_dict.get("energy_level"),
            emotion=analysis_dict.get("emotion"),
            formality=analysis_dict.get("formality"),
            tone=analysis_dict.get("tone"),
            language=(analysis_dict.get("metadata", {}) or {}).get("detected_language", "en"),
        )
        return self.selector.select(c)


def get_voice_catalog() -> Dict[str, VoiceProfile]:
    return VOICE_CATALOG.copy()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("=" * 80)
    print("VOICE SELECTOR DEMO")
    print("=" * 80)

    selector = VoiceSelector()

    print("\n== Test 1: Modern Selection Strategy ==")
    tests_modern = [
        {"voice_type": "warm", "energy_level": "calm", "expected": "Rachel"},
        {"voice_type": "authoritative", "energy_level": "calm", "expected": "Antoni"},
        {"voice_type": "dynamic", "energy_level": "energetic", "expected": "Adam"},
        {"voice_type": "gentle", "energy_level": "calm", "emotion": "melancholic", "expected": "Elli"},
    ]
    for i, case in enumerate(tests_modern, 1):
        expected = case.pop("expected")
        c = VoiceSelectionCriteria(
            voice_type=case.get("voice_type"),
            energy_level=case.get("energy_level"),
            emotion=case.get("emotion"),
            language=case.get("language", "en"),
        )
        v = selector.select(c)
        status = "OK" if v.name == expected else "FAIL"
        print(f"{status} Test {i}: {case}")
        print(f"   Selected: {v.name} (expected: {expected})")
        print(f"   Voice ID: {v.voice_id}")

    print("\n== Test 2: Legacy Compatibility ==")
    tests_legacy = [
        {"gender": "male", "age": "adult", "tone": "serious", "expected": "Antoni"},
        {"gender": "male", "age": "young", "tone": "energetic", "expected": "Adam"},
        {"gender": "female", "age": "adult", "tone": "warm", "expected": "Rachel"},
        {"gender": "female", "age": "young", "tone": "emotional", "expected": "Elli"},
    ]
    for i, case in enumerate(tests_legacy, 1):
        expected = case.pop("expected")
        c = VoiceSelectionCriteria(
            tone=case.get("tone"),
            gender=case.get("gender"),
            age=case.get("age"),
            language=case.get("language", "en"),
        )
        v = selector.select(c)
        status = "OK" if v.name == expected else "FAIL"
        print(f"{status} Test {i}: {case}")
        print(f"   Selected: {v.name} (expected: {expected})")

    print("\n== Test 3: Voice Catalog Listing ==")
    print("All available voices:")
    for name, profile in VOICE_CATALOG.items():
        print(f"  - {name}: {profile.description[:60]}...")
        print(f"    Gender: {profile.gender.value}, Age: {profile.age.value}")
        print(f"    Tones: {', '.join([t.value for t in profile.primary_tones])}")

    print("\n== Test 4: QuoteAnalyzer Integration ==")
    mock = {
        "voice_type": "warm",
        "energy_level": "balanced",
        "emotion": "hopeful",
        "formality": "informal",
        "tone": "hopeful",
        "metadata": {"detected_language": "en"},
    }
    integration_selector = VoiceSelectorForQuoteAnalyzer()
    v = integration_selector.select_from_dict(mock)
    print(f"Analysis: {mock}")
    print(f"Selected Voice: {v.name}")
    print(f"   Voice ID: {v.voice_id}")
    print(f"   Description: {v.description}")

    print("\n" + "=" * 80)
    print("All tests complete!")

