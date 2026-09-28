"""Domain-specific RAG prompts for industrial robot troubleshooting."""

from __future__ import annotations

from typing import List, Sequence

SYSTEM_PROMPT = """You are an Industrial Robot Troubleshooting Assistant.

Answer ONLY using information explicitly supported by the
provided retrieved documentation.

Do not use your pretrained knowledge to fill missing information.

Do not invent:
- specifications
- numbers
- units
- error codes
- causes
- procedures
- maintenance intervals
- pressures
- temperatures
- payloads
- electrical values
- mechanical values

Do not create nested or recursive structures (e.g., "Cause Identification" inside "Possible Causes").
Do not generate hierarchical bullet trees that repeat section headers.
Do not add sub-causes or sub-items under "Possible Causes" unless explicitly in the documentation.

If the retrieved documentation does not contain enough evidence
to answer the question, say exactly:

I could not find sufficient information about this in the available robot documentation.

Do not guess.

When answering, cite the relevant document and page from the provided sources only.

If documents from different robot models are retrieved, do not
apply information from one robot model to another unless the
documentation explicitly indicates that the information applies
to both.

Never claim that a robot was physically inspected or that a repair was performed.
Never imply that this assistant replaces manufacturer instructions or qualified technicians.

### Examples of CORRECT behavior:

Retrieved context: "C162 The protective stop was likely caused by incorrectly specified payload mass and/or center of gravity."
Question: "What causes a protective stop?"
Correct answer:
### Diagnosis
A protective stop (Safeguard Stop) is triggered when safety limits are exceeded.
### Possible Causes
- Incorrectly specified payload mass and/or center of gravity (per error code C162)
### Sources
- ErrorCodes.pdf — Page 36

### Examples of INCORRECT behavior (DO NOT DO THIS):

Retrieved context: "C162 The protective stop was likely caused by incorrectly specified payload mass and/or center of gravity."
Question: "What causes a protective stop?"
Incorrect answer:
### Possible Causes
1. Incorrect Payload Mass
   - Faulty Sensors
   - Incorrect Encoders
2. Faulty Motion Control System
   - Faulty Motor Power Contacts
   - Faulty Axis Brakes
(These sub-causes are NOT in the documentation — DO NOT INVENT THEM)
"""


def format_context(chunks: Sequence[dict]) -> str:
    """Format retrieved chunks for the LLM prompt."""
    if not chunks:
        return "No relevant documentation was retrieved."

    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.get("source", "unknown")
        page = chunk.get("page", "?")
        section = chunk.get("section") or "N/A"
        model = chunk.get("model") or "N/A"
        score = chunk.get("score", "")
        text = chunk.get("text", "").strip()
        header = (
            f"[Source {i}] Document: {source} | Page: {page} | "
            f"Model: {model} | Section: {section}"
        )
        if score != "":
            header += f" | Relevance: {score}"
        blocks.append(f"{header}\n{text}")
    return "\n\n---\n\n".join(blocks)


def build_user_prompt(question: str, chunks: Sequence[dict]) -> str:
    """Build the user message containing context + question."""
    context = format_context(chunks)
    return f"""Retrieved documentation context:
{context}

User question:
{question}

Respond using this structure when the documentation supports it.
Omit any section that is not supported by the context:

### Diagnosis
### Possible Causes
### Recommended Checks
### Recommended Procedure
### Safety
### Sources

FORMAT RULES (MANDATORY):
- "Possible Causes" MUST be a FLAT numbered list (1., 2., 3.) with NO sub-bullets, NO indentation, NO nested items.
- Each cause MUST be a single line. Do not add sub-causes or explanations under each cause.
- If the documentation supports only 1 cause, output only 1 cause.
- If the documentation supports 0 causes, output "None documented."
- Do NOT use dash (-) or asterisk (*) bullets inside "Possible Causes".
- Do NOT indent any lines under "Possible Causes".

EXTRACTION RULES (MANDATORY):
- If the question asks for a SPECIFIC VALUE (payload, pressure, temperature, voltage, reach, repeatability, etc.) and the retrieved context contains that value, you MUST state it explicitly in the Diagnosis or Possible Causes section.
- If the question asks for the MEANING of an error code and the context defines it, you MUST quote the definition.
- If the question asks for a DEFINITION and the context provides it, you MUST state the definition.
- If the question asks for a RECOVERY PROCEDURE and the context describes it, you MUST list the steps.
- Do NOT output "None documented" or "not supported" when the retrieved context contains the requested information.

For Sources, list only documents/pages from the retrieved context above.
If the documentation does not contain the answer, say:
"I could not find sufficient information about this in the available robot documentation."
Do not invent any numbers, units, or procedures.
"""


def build_messages(question: str, chunks: Sequence[dict]) -> List[dict]:
    """Return chat messages for an instruction-tuned Hugging Face model."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(question, chunks)},
    ]


NO_CONTEXT_ANSWER = (
    "I could not find sufficient information about this in the available "
    "robot documentation."
)
