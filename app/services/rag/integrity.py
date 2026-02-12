import re
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

@dataclass
class ValidationIssue:
    level: int
    header: str
    issue_type: str
    message: str

class StructureValidator:
    PATTERNS = {
        "decimal": re.compile(r"^(\d+(?:\.\d+)*)"),
        "roman_upper": re.compile(r"^([IVXLCDM]+)\."),
        "alpha_upper": re.compile(r"^([A-Z])\."),
        "alpha_lower": re.compile(r"^([a-z])\."),
        "paren_num": re.compile(r"^\((\d+)\)"),
    }

    def __init__(self):
        pass

    def _roman_to_int(self, s: str) -> int:
        rom_val = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        int_val = 0
        for i in range(len(s)):
            if i > 0 and rom_val[s[i]] > rom_val[s[i - 1]]:
                int_val += rom_val[s[i]] - 2 * rom_val[s[i - 1]]
            else:
                int_val += rom_val[s[i]]
        return int_val

    def _alpha_to_int(self, s: str) -> int:
        if not s: return 0
        return ord(s.upper()) - 64

    def _parse_number(self, text: str, pattern_type: str) -> Optional[int]:
        if pattern_type == "decimal":
            match = self.PATTERNS["decimal"].match(text)
            if match:
                parts = match.group(1).split('.')
                return int(parts[-1])

        elif pattern_type == "roman_upper":
            match = self.PATTERNS["roman_upper"].match(text)
            if match:
                return self._roman_to_int(match.group(1))

        elif pattern_type in ["alpha_upper", "alpha_lower"]:
            match = self.PATTERNS[pattern_type].match(text)
            if match:
                return self._alpha_to_int(match.group(1))

        elif pattern_type == "paren_num":
            match = self.PATTERNS["paren_num"].match(text)
            if match:
                return int(match.group(1))

        return None

    def _detect_pattern(self, header_text: str) -> Optional[str]:
        header_text = header_text.strip()
        for name, pattern in self.PATTERNS.items():
            if pattern.match(header_text):
                return name
        return None

    def _parse_line(self, line: str) -> Optional[Tuple[int, str]]:
        # A. Markdown Headers
        match_header = re.match(r'^(#{1,6})\s+(.+)$', line)
        if match_header:
            level = len(match_header.group(1))
            content = match_header.group(2).strip()
            return level, content

        # B. List Items (e.g., "1. Text", "  a. Text")
        match_list = re.match(r'^(\s*)((?:\d+\.)+|[a-zA-Z]\.|[IVXLCDM]+\.)\s+(.+)$', line)
        if match_list:
            indent = match_list.group(1)
            marker = match_list.group(2)
            content = match_list.group(3).strip()

            # Determine level
            if '.' in marker[:-1] and marker[0].isdigit():
                level = marker.count('.')
            else:
                level = (len(indent) // 2) + 1

            level = min(level, 6)
            return level, f"{marker} {content}"

        return None

    def _analyze_structure(self, headers: List[Tuple[int, str]]) -> List[ValidationIssue]:
        issues = []
        last_numbers: Dict[int, int] = {}
        level_patterns: Dict[int, str] = {}
        prev_level = 0

        for level, text in headers:
            # Hierarchy Check
            if level > prev_level + 1 and prev_level != 0:
                 pass

            prev_level = level
            pattern = self._detect_pattern(text)

            if level not in level_patterns:
                if pattern:
                    level_patterns[level] = pattern

            current_pattern = level_patterns.get(level)

            if current_pattern and pattern == current_pattern:
                num = self._parse_number(text, current_pattern)
                if num is not None:
                    last_num = last_numbers.get(level, 0)
                    expected = last_num + 1

                    if num > expected and not (expected == 1 and num == 1):
                        issues.append(ValidationIssue(
                            level=level,
                            header=text,
                            issue_type="gap",
                            message=f"Missing item? Expected {expected}, found {num}."
                        ))

                    last_numbers[level] = num
                    for l in range(level + 1, 7):
                        last_numbers[l] = 0

            elif current_pattern and not pattern:
                pass

        return issues

    def validate(self, text: str) -> List[ValidationIssue]:
        headers = []
        lines = text.split('\n')

        for line in lines:
            parsed = self._parse_line(line)
            if parsed:
                headers.append(parsed)

        return self._analyze_structure(headers)

validator = StructureValidator()
