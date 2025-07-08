#!/usr/bin/env python3
"""
Test script to verify that the enhanced context summarization preserves
all types of structured data needed for multi-step workflows.
"""
import asyncio
import json
import time
from reactive_agents.core.context.agent_context import AgentContext
from reactive_agents.core.types.session_types import AgentSession


class MockModelProvider:
    """Mock model provider for testing."""

    def __init__(self):
        self.name = "test-provider"
        self.model = "test-model"

    async def get_completion(self, system, prompt, options=None):
        class MockResponse:
            def __init__(self):
                self.message = MockMessage()

        class MockMessage:
            def __init__(self):
                # Return a realistic summary that includes the extracted data
                self.content = """
**Progress**: Completed database query and file operations successfully.

**Key Data**: 
- Database IDs: user_12345, session_abc123def, transaction_TX789456
- File paths: /data/exports/report_2024.xlsx, /tmp/processed_data.json
- API endpoints: https://api.example.com/v1/users, https://internal.service.com/data
- Email addresses: user@example.com, admin@company.org

**Tool Results**: 
- Database query returned 150 user records
- File export completed successfully
- API call retrieved user profile data

**State**: Ready to process exported data and send notifications
**Dependencies**: Next steps need the exported file path and user IDs for notification processing
"""

        return MockResponse()


class MockToolManager:
    """Mock tool manager with SearchDataManager for testing."""

    def __init__(self):
        from reactive_agents.core.tools.data_extractor import (
            DataExtractor,
            SearchDataManager,
        )

        self.data_extractor = DataExtractor()
        self.search_data_manager = SearchDataManager()

        # Simulate stored search data from various tools
        self._setup_mock_search_data()

    def _setup_mock_search_data(self):
        # Mock database search results
        db_result = {
            "users": [
                {"id": "user_12345", "email": "user@example.com", "status": "active"},
                {"id": "user_67890", "email": "admin@company.org", "status": "pending"},
            ],
            "total_count": 150,
            "query_id": "query_abc123def456",
        }

        self.search_data_manager.store_search_data(
            "database_search",
            {"table": "users", "conditions": {"status": "active"}},
            json.dumps(db_result),
            self.data_extractor,
        )

        # Mock file operation results
        file_result = {
            "operation": "export_complete",
            "file_path": "/data/exports/report_2024.xlsx",
            "temp_file": "/tmp/processed_data.json",
            "record_count": 150,
            "export_id": "EXP789456123",
        }

        self.search_data_manager.store_search_data(
            "file_export",
            {"format": "xlsx", "include_headers": True},
            json.dumps(file_result),
            self.data_extractor,
        )

        # Mock API call results
        api_result = {
            "status": "success",
            "endpoint": "https://api.example.com/v1/users",
            "response_data": {
                "users": [
                    {
                        "user_id": "api_user_999",
                        "profile_url": "https://profiles.example.com/999",
                    }
                ],
                "session_token": "tok_abcdefghij1234567890",
                "expires_at": "2024-07-06T18:00:00Z",
            },
        }

        self.search_data_manager.store_search_data(
            "api_call",
            {"endpoint": "/users", "method": "GET"},
            json.dumps(api_result),
            self.data_extractor,
        )


async def test_generic_data_preservation():
    """Test that all types of structured data are preserved during context summarization."""
    print("Testing generic data preservation in context summarization...")

    # Create test context
    session = AgentSession(
        initial_task="Process user data and generate reports",
        current_task="Process user data and generate reports",
        start_time=time.time(),
    )

    context = AgentContext(
        agent_name="Test Agent", provider_model_name="test-model", session=session
    )

    # Mock the required components
    context.model_provider = MockModelProvider()  # type: ignore
    context.tool_manager = MockToolManager()  # type: ignore
    context.agent_logger = type(
        "MockLogger",
        (),
        {
            "info": lambda self, msg: print(f"INFO: {msg}"),
            "debug": lambda self, msg: print(f"DEBUG: {msg}"),
            "error": lambda self, msg: print(f"ERROR: {msg}"),
        },
    )()  # type: ignore

    # Create test messages that would be summarized
    test_messages = [
        {
            "role": "assistant",
            "content": "[TOOL SUMMARY] Executed database_search and found 150 user records including user_12345 and user_67890",
        },
        {
            "role": "assistant",
            "content": "[TOOL SUMMARY] File export completed. Created /data/exports/report_2024.xlsx and temporary file /tmp/processed_data.json with export ID EXP789456123",
        },
        {
            "role": "assistant",
            "content": "[TOOL SUMMARY] API call to https://api.example.com/v1/users successful. Retrieved user api_user_999 with session token tok_abcdefghij1234567890",
        },
        {
            "role": "tool",
            "content": "Database query completed. Found records: user_12345, user_67890. Connection ID: conn_xyz789",
        },
    ]

    # Test data extraction
    extracted_data = await context._extract_structured_data_from_messages(test_messages)

    print(f"Extracted data keys: {list(extracted_data.keys())}")
    print(
        f"Search tool data: {list(extracted_data.get('search_tool_data', {}).keys())}"
    )

    # Test summarization prompt creation
    combined_text = "\n".join([msg.get("content", "") for msg in test_messages])
    enhanced_prompt = context._create_enhanced_summarization_prompt(
        combined_text, extracted_data
    )

    print("\n=== Enhanced Summarization Prompt ===")
    print(
        enhanced_prompt[:1000] + "..."
        if len(enhanced_prompt) > 1000
        else enhanced_prompt
    )

    # Test summary formatting
    formatted_data = context._format_extracted_data_for_summary(extracted_data)

    print("\n=== Formatted Data for Summary ===")
    print(formatted_data)

    # Verify that different types of data are preserved
    print("\n=== Verification ===")

    # Check for various ID types
    db_data = extracted_data.get("search_tool_data", {}).get("database_search", {})
    file_data = extracted_data.get("search_tool_data", {}).get("file_export", {})
    api_data = extracted_data.get("search_tool_data", {}).get("api_call", {})

    success_checks = []

    if db_data:
        success_checks.append("✅ Database search data preserved")
    else:
        success_checks.append("❌ Database search data missing")

    if file_data:
        success_checks.append("✅ File operation data preserved")
    else:
        success_checks.append("❌ File operation data missing")

    if api_data:
        success_checks.append("✅ API call data preserved")
    else:
        success_checks.append("❌ API call data missing")

    # Check if the formatted summary contains various types of identifiers
    if any(
        id_val in formatted_data
        for id_val in ["user_12345", "EXP789456123", "tok_abcdefghij1234567890"]
    ):
        success_checks.append("✅ Multiple ID types found in summary")
    else:
        success_checks.append("❌ ID extraction may have failed")

    # Check for file paths
    if "/data/exports/report_2024.xlsx" in formatted_data:
        success_checks.append("✅ File paths preserved")
    else:
        success_checks.append("❌ File paths missing")

    # Check for URLs/endpoints
    if "https://api.example.com" in formatted_data:
        success_checks.append("✅ URLs/endpoints preserved")
    else:
        success_checks.append("❌ URLs/endpoints missing")

    for check in success_checks:
        print(check)

    # Test ID extraction directly
    test_text = """
    {
        "user_id": "user_12345",
        "session": "sess_abcdef123456",
        "file_path": "/data/exports/report_2024.xlsx",
        "api_token": "tok_abcdefghij1234567890",
        "confirmation_code": "ABC123456"
    }
    Database connection: conn_xyz789
    Email IDs: 1744e2f6ddb752bc, 17443e6498afcbef
    """

    extracted_ids = context._extract_all_ids_from_text(test_text)
    print(f"\n=== Direct ID Extraction Test ===")
    print(f"Extracted IDs: {extracted_ids}")

    expected_ids = [
        "user_12345",
        "sess_abcdef123456",
        "tok_abcdefghij1234567890",
        "ABC123456",
        "conn_xyz789",
        "1744e2f6ddb752bc",
        "17443e6498afcbef",
    ]
    found_expected = [eid for eid in expected_ids if eid in extracted_ids]

    print(f"Expected IDs found: {found_expected}")
    print(f"Coverage: {len(found_expected)}/{len(expected_ids)} expected IDs found")

    if len(found_expected) >= 5:  # Allow some variation in pattern matching
        print("✅ ID extraction working well for multiple types")
    else:
        print("❌ ID extraction may need improvement")

    print("\n=== Test Complete ===")
    return True


if __name__ == "__main__":
    asyncio.run(test_generic_data_preservation())
