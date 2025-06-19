"""
Data extraction utilities for processing tool results and search data.
"""

import re
import json
import time
from typing import Dict, Any, List, Optional
from pydantic import BaseModel


class ExtractedData(BaseModel):
    """Model for structured extracted data."""

    prices: List[str] = []
    percentages: List[str] = []
    dates: List[str] = []
    times: List[str] = []
    numbers: List[str] = []
    emails: List[str] = []
    urls: List[str] = []
    phone_numbers: List[str] = []
    entities: List[str] = []
    structured_data: Dict[str, Any] = {}


class DataExtractor:
    """Extracts structured data from text content."""

    def __init__(self):
        # Generic data patterns
        self.patterns = {
            "prices": [
                r"(\$[\d,]+\.?\d*)",  # $1,234.56 or $1234
                r"(\d+\.?\d*\s*USD)",  # 1234.56 USD
                r"(\d+\.?\d*\s*\$)",  # 1234.56 $
                r"(\d+\.?\d*\s*EUR)",  # 1234.56 EUR
                r"(\d+\.?\d*\s*GBP)",  # 1234.56 GBP
                r"(\d+\.?\d*\s*JPY)",  # 1234.56 JPY
            ],
            "percentages": [
                r"(\d+\.?\d*%)",  # 12.5%
                r"(\d+\.?\d*\s*percent)",  # 12.5 percent
            ],
            "dates": [
                r"(\d{1,2}/\d{1,2}/\d{2,4})",  # MM/DD/YYYY or M/D/YY
                r"(\d{4}-\d{2}-\d{2})",  # YYYY-MM-DD
                r"(\w+\s+\d{1,2},?\s+\d{4})",  # January 15, 2024
                r"(\d{1,2}\s+\w+\s+\d{4})",  # 15 January 2024
            ],
            "times": [
                r"(\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?)",  # 14:30 or 2:30 PM
                r"(\d{1,2}:\d{2}(?::\d{2})?\s*(?:UTC|GMT|EST|PST))",  # 14:30 UTC
            ],
            "numbers": [
                r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?)",  # 1,234,567.89
                r"(\d+\.?\d*)",  # Any number
            ],
            "emails": [
                r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
            ],
            "urls": [
                r"(https?://[^\s]+)",
                r"(www\.[^\s]+)",
            ],
            "phone_numbers": [
                r"(\+?1?\s*\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4})",  # US phone numbers
                r"(\+\d{1,3}\s?\d{1,4}\s?\d{1,4}\s?\d{1,4})",  # International
            ],
        }

        # Entity patterns for named entity recognition
        self.entity_patterns = [
            # Companies and organizations
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|Corp|LLC|Ltd|Company|Corporation|Organization)\b",
            # Product names
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Pro|Max|Ultra|Plus|Elite|Premium)\b",
            # Technology terms
            r"\b(?:AI|ML|API|SDK|CPU|GPU|RAM|SSD|HDD|USB|WiFi|Bluetooth|5G|4G|LTE)\b",
            # Financial terms
            r"\b(?:Bitcoin|Ethereum|XRP|Solana|Cardano|BTC|ETH|USD|EUR|GBP|JPY|CNY|Market Cap|Volume|Price)\b",
            # Location patterns
            r"\b(?:New York|London|Tokyo|Paris|Berlin|Sydney|Toronto|Mumbai|Beijing|Moscow)\b",
            # Person names (simple pattern)
            r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b",
        ]

    def extract_all(self, text: str) -> ExtractedData:
        """Extract all types of data from text."""
        extracted = ExtractedData()

        # Extract data for each pattern type
        for data_type, pattern_list in self.patterns.items():
            matches = []
            for pattern in pattern_list:
                found = re.findall(pattern, text, re.IGNORECASE)
                matches.extend(found)

            if matches:
                # Remove duplicates while preserving order
                unique_matches = []
                for match in matches:
                    if match not in unique_matches:
                        unique_matches.append(match)
                setattr(
                    extracted, data_type, unique_matches[:10]
                )  # Limit to first 10 matches

        # Extract named entities and key terms
        extracted.entities = self._extract_entities(text)

        # Extract structured data (JSON-like patterns)
        extracted.structured_data = self._extract_structured_data(text)

        return extracted

    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities and key terms from text."""
        entities = []

        for pattern in self.entity_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities.extend(matches)

        # Remove duplicates
        return list(set(entities))[:20]  # Limit to 20 entities

    def _extract_structured_data(self, text: str) -> Dict[str, Any]:
        """Extract structured data patterns from text."""
        structured = {}

        # Try to find JSON-like structures
        json_patterns = [
            r"\{[^{}]*\}",  # Simple JSON objects
            r"\[[^\[\]]*\]",  # Simple JSON arrays
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    parsed = json.loads(match)
                    if isinstance(parsed, dict):
                        structured.update(parsed)
                    elif isinstance(parsed, list):
                        structured["list_data"] = parsed
                except json.JSONDecodeError:
                    continue

        # Extract key-value pairs
        kv_patterns = [
            r"(\w+):\s*([^,\n]+)",  # key: value
            r"(\w+)\s*=\s*([^,\n]+)",  # key = value
        ]

        for pattern in kv_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for key, value in matches:
                key = key.strip().lower()
                value = value.strip()
                if value and len(value) < 100:  # Avoid very long values
                    structured[key] = value

        return structured

    def extract_specific_type(self, text: str, data_type: str) -> List[str]:
        """Extract a specific type of data from text."""
        if data_type not in self.patterns:
            return []

        matches = []
        for pattern in self.patterns[data_type]:
            found = re.findall(pattern, text, re.IGNORECASE)
            matches.extend(found)

        # Remove duplicates while preserving order
        unique_matches = []
        for match in matches:
            if match not in unique_matches:
                unique_matches.append(match)

        return unique_matches[:10]  # Limit to first 10 matches

    def validate_data_usage(
        self, content: str, extracted_data: ExtractedData
    ) -> Dict[str, Any]:
        """Validate that extracted data is being used in content."""
        validation = {"valid": True, "warnings": [], "suggestions": [], "data_used": []}

        # Check if any extracted data is used in the content
        data_used = False
        for field_name, values in extracted_data.dict().items():
            if field_name == "structured_data":
                continue

            if isinstance(values, list):
                for value in values:
                    if str(value) in content:
                        data_used = True
                        validation["data_used"].append(f"{field_name}: {value}")
                        validation["suggestions"].append(f"Using {field_name}: {value}")
                        break
            elif str(values) in content:
                data_used = True
                validation["data_used"].append(f"{field_name}: {values}")
                validation["suggestions"].append(f"Using {field_name}: {values}")

        if not data_used and any(extracted_data.dict().values()):
            validation["warnings"].append("Content doesn't match found extracted data")
            validation["suggestions"].append(
                f"Consider using found data: {extracted_data.dict()}"
            )
            validation["valid"] = False

        return validation


class SearchDataManager:
    """Manages search data storage and retrieval."""

    def __init__(self):
        self.search_data: Dict[str, Dict[str, Any]] = {}

    def store_search_data(
        self,
        tool_name: str,
        params: Dict[str, Any],
        result: Any,
        extractor: DataExtractor,
    ) -> None:
        """Store structured data from search tools."""
        try:
            result_str = str(result)

            # Store the raw result and parameters for context
            self.search_data[tool_name] = {
                "params": params,
                "result": result_str,
                "timestamp": time.time(),
                "extracted_data": extractor.extract_all(result_str).dict(),
            }

        except Exception as e:
            # Log error but don't fail
            print(f"Failed to store search data for {tool_name}: {e}")

    def get_search_data(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get stored search data for a tool."""
        return self.search_data.get(tool_name)

    def get_all_search_data(self) -> Dict[str, Dict[str, Any]]:
        """Get all stored search data."""
        return self.search_data.copy()

    def clear_search_data(self) -> None:
        """Clear all stored search data."""
        self.search_data.clear()

    def get_extracted_data(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get extracted data for a specific tool."""
        search_data = self.get_search_data(tool_name)
        if search_data:
            return search_data.get("extracted_data", {})
        return None
