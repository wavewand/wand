#!/usr/bin/env python3
"""
Simple test script to verify the MCP system is working.
"""

import json
import time

import requests


def test_system():
    """Simple test of the MCP system."""
    base_url = "http://localhost:8000"

    print("🧪 Testing MCP Distributed System")
    print("=" * 40)

    try:
        # Test health endpoint
        print("1. Testing health check...")
        response = requests.get(f"{base_url}/health", timeout=5)
        health = response.json()
        print(f"   Status: {health['status']}")

        # Test system status
        print("\n2. Testing system status...")
        response = requests.get(f"{base_url}/api/v1/system/status", timeout=5)
        status = response.json()
        print(f"   Agents: {status['active_agents']}/{status['total_agents']}")
        print(f"   Tasks: {status['total_tasks']}")

        # Test task creation
        print("\n3. Testing task creation...")
        task_data = {"title": "Test task", "description": "Simple test task", "type": "backend", "priority": "medium"}
        response = requests.post(f"{base_url}/api/v1/tasks", json=task_data, timeout=10)
        task_result = response.json()
        print(f"   Task created: {task_result['task_id']}")
        print(f"   Status: {task_result['status']}")

        # Test integration
        print("\n4. Testing Slack integration...")
        slack_data = {
            "operation": "send_message",
            "parameters": {"channel": "#test", "message": "Hello from MCP system test!"},
        }
        response = requests.post(f"{base_url}/api/v1/integrations/slack", json=slack_data, timeout=10)
        slack_result = response.json()
        print(f"   Success: {slack_result['success']}")
        print(f"   Message: {slack_result['message']}")

        print("\n✅ All tests passed!")

    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to MCP system.")
        print("   Make sure it's running: python main.py")
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    test_system()
