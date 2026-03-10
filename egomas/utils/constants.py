"""
Shared constants for EgoMAS (MA-EgoQA) pipeline.
"""
import re

# Person names used in planner selection
PERSON_NAMES = ["Jake", "Alice", "Katrina", "Lucia", "Tasha", "Shure"]

# Valid option letters for multiple-choice answers (A-E)
VALID_OPTIONS = ["a", "b", "c", "d", "e"]

# Regex to extract content from ```json or ``` code blocks
CODEBLOCK_PATTERN = re.compile(r"```(?:json|python)?\s*(.*?)```", re.DOTALL)
