from __future__ import annotations

from dataclasses import asdict, dataclass


MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
PROMPT_VARIANTS = ("direct", "controlled")
SUPPORTED_SHOTS = (0, 3)
EXPERIMENT_SETTINGS = (
    "direct_0shot",
    "controlled_0shot",
    "direct_3shot",
    "controlled_3shot",
)
MAX_NEW_TOKENS = 96
MAX_INPUT_LENGTH = 1024
FEW_SHOT_ROW_NUMBERS = (1, 2, 6)


@dataclass(frozen=True)
class FewShotExample:
    row_number: int
    source: str
    target: str


# These three examples come only from the deterministically shuffled
# WikiAuto training split (seed 468, shuffle buffer 10,000). They are
# fixed here so the exact few-shot prompts can always be reproduced.
FEW_SHOT_EXAMPLES = (
    FewShotExample(
        row_number=1,
        source=(
            "Milliken was born in Traverse City , Michigan , the "
            "second child in a family familiar with the intricacies "
            "of public service ."
        ),
        target=(
            "Milliken was born in Traverse City , Michigan ."
        ),
    ),
    FewShotExample(
        row_number=2,
        source=(
            "The 1966 tornado was significantly stronger than the "
            "other ten tornadoes that struck Topeka prior to June 8 ."
        ),
        target=(
            "The 1966 tornado was much stronger than the other ten "
            "tornadoes that hit Topeka before June 8 ."
        ),
    ),
    FewShotExample(
        row_number=6,
        source=(
            "The Pistol Star was discovered using the Hubble Space "
            "Telescope in the early 1990s by Don Figer , an "
            "astronomer at UCLA ."
        ),
        target=(
            "The Pistol Star was discovered by the Hubble Space "
            "Telescope in the early 1990s ."
        ),
    ),
)


def system_prompt(variant: str) -> str:
    if variant == "direct":
        return (
            "You are an English text simplification assistant."
        )

    if variant == "controlled":
        return (
            "You rewrite complex English so it is clear to a "
            "general reader."
        )

    raise ValueError(
        f"Unknown prompt variant: {variant!r}. "
        f"Choose from {PROMPT_VARIANTS}."
    )


def user_prompt(text: str, variant: str) -> str:
    text = text.strip()

    if not text:
        raise ValueError("The source text cannot be empty.")

    if variant == "direct":
        return (
            "Simplify the following sentence. Keep the original "
            "meaning and important facts. Use easier words and "
            "shorter sentence structure. Return only the simplified "
            "text.\n\n"
            f"Complex sentence: {text}\n"
            "Simplified sentence:"
        )

    if variant == "controlled":
        return (
            "Rewrite the text below under these rules:\n"
            "1. Preserve names, numbers, dates, and important facts.\n"
            "2. Do not add information.\n"
            "3. Use common words and split long sentences when "
            "helpful.\n"
            "4. Return only the rewritten text, without commentary."
            "\n\n"
            f"Text: {text}\n"
            "Rewrite:"
        )

    raise ValueError(
        f"Unknown prompt variant: {variant!r}. "
        f"Choose from {PROMPT_VARIANTS}."
    )


def setting_name(variant: str, shots: int) -> str:
    if variant not in PROMPT_VARIANTS:
        raise ValueError(
            f"Unknown prompt variant: {variant!r}."
        )

    if shots not in SUPPORTED_SHOTS:
        raise ValueError(
            f"Unsupported shot count: {shots}. "
            f"Choose from {SUPPORTED_SHOTS}."
        )

    return f"{variant}_{shots}shot"


def parse_setting(setting: str) -> tuple[str, int]:
    for variant in PROMPT_VARIANTS:
        for shots in SUPPORTED_SHOTS:
            if setting == setting_name(variant, shots):
                return variant, shots

    raise ValueError(
        f"Unknown experiment setting: {setting!r}. "
        f"Choose from {EXPERIMENT_SETTINGS}."
    )


def build_messages(
    source: str,
    variant: str,
    shots: int,
) -> list[dict[str, str]]:
    if shots not in SUPPORTED_SHOTS:
        raise ValueError(
            f"Unsupported shot count: {shots}. "
            f"Choose from {SUPPORTED_SHOTS}."
        )

    messages = [
        {
            "role": "system",
            "content": system_prompt(variant),
        }
    ]

    for example in FEW_SHOT_EXAMPLES[:shots]:
        messages.extend(
            [
                {
                    "role": "user",
                    "content": user_prompt(
                        example.source,
                        variant,
                    ),
                },
                {
                    "role": "assistant",
                    "content": example.target,
                },
            ]
        )

    messages.append(
        {
            "role": "user",
            "content": user_prompt(source, variant),
        }
    )

    return messages


def prompt_record(
    variant: str,
    shots: int,
) -> dict[str, object]:
    """Return the exact prompt specification saved with results."""
    setting = setting_name(variant, shots)

    return {
        "setting": setting,
        "variant": variant,
        "shots": shots,
        "system_prompt": system_prompt(variant),
        "user_prompt_template": user_prompt(
            "{source}",
            variant,
        ),
        "few_shot_row_numbers": list(
            FEW_SHOT_ROW_NUMBERS[:shots]
        ),
        "few_shot_examples": [
            asdict(example)
            for example in FEW_SHOT_EXAMPLES[:shots]
        ],
    }
