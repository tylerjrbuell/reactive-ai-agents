"""Tool usage guard and policy enforcement."""

import time
from typing import Dict, List, Set, Tuple


class ToolGuard:
    """Middleware for enforcing tool usage policies (rate limits, cooldowns, confirmation, etc)."""

    def __init__(self):
        self.usage_log: Dict[str, List[float]] = {}  # {tool_name: [timestamps]}
        self.rate_limits: Dict[str, Tuple[int, int]] = {}  # {tool_name: (max_calls, per_seconds)}
        self.confirmation_required: Set[str] = set()
        self.admin_required: Set[str] = set()
        self.cooldowns: Dict[str, int] = {}  # {tool_name: cooldown_seconds}

    def add_default_guards(self):
        """Add default guard policies for common sensitive tools."""
        # Rate limits: (max_calls, per_seconds)
        self.rate_limits.update(
            {
                # Email operations
                "write_email": (1, 60),  # 1 email per 60 seconds
                "send_email": (1, 60),  # 1 email per 60 seconds
                "compose_email": (1, 60),  # 1 email per 60 seconds
                # File operations
                "delete_file": (5, 300),  # 5 deletions per 5 minutes
                "move_file": (10, 300),  # 10 moves per 5 minutes
                "copy_file": (10, 300),  # 10 copies per 5 minutes
                # Database operations
                "delete_record": (10, 300),  # 10 deletions per 5 minutes
                "update_record": (20, 300),  # 20 updates per 5 minutes
                "drop_table": (1, 3600),  # 1 table drop per hour
                "truncate_table": (1, 3600),  # 1 truncate per hour
                # API operations
                "api_call": (30, 60),  # 30 API calls per minute
                "http_request": (30, 60),  # 30 HTTP requests per minute
                "webhook": (10, 300),  # 10 webhooks per 5 minutes
                # System operations
                "execute_command": (5, 300),  # 5 commands per 5 minutes
                "run_script": (3, 600),  # 3 scripts per 10 minutes
                "install_package": (1, 3600),  # 1 package install per hour
                # Financial operations
                "make_payment": (1, 3600),  # 1 payment per hour
                "transfer_money": (1, 3600),  # 1 transfer per hour
                "create_invoice": (5, 300),  # 5 invoices per 5 minutes
                # User management
                "create_user": (3, 600),  # 3 users per 10 minutes
                "delete_user": (1, 3600),  # 1 user deletion per hour
                "change_password": (5, 300),  # 5 password changes per 5 minutes
                # Email management (Gmail specific)
                "trash_emails": (10, 300),  # 10 trash operations per 5 minutes
                "archive_emails": (20, 300),  # 20 archive operations per 5 minutes
                "star_emails": (30, 300),  # 30 star operations per 5 minutes
            }
        )

        # Tools requiring explicit confirmation
        self.confirmation_required.update(
            {
                # Email operations
                "write_email",
                "send_email",
                "compose_email",
                # Destructive file operations
                "delete_file",
                "delete_directory",
                "format_drive",
                # Database operations
                "delete_record",
                "drop_table",
                "truncate_table",
                "delete_database",
                # System operations
                "execute_command",
                "run_script",
                "install_package",
                "uninstall_package",
                "restart_service",
                "stop_service",
                "kill_process",
                # Financial operations
                "make_payment",
                "transfer_money",
                "create_invoice",
                "refund_payment",
                # User management
                "delete_user",
                "change_password",
                "reset_password",
                "grant_admin",
                # Network operations
                "open_port",
                "close_port",
                "block_ip",
                "unblock_ip",
                # Email management (bulk operations)
                "trash_emails",
                "archive_emails",
                "delete_emails",
            }
        )

        # Tools requiring admin privileges (additional logging)
        self.admin_required.update(
            {
                "drop_table",
                "delete_database",
                "format_drive",
                "kill_process",
                "grant_admin",
                "open_port",
                "close_port",
                "block_ip",
                "unblock_ip",
            }
        )

        # Tools with cooldown periods (minimum time between uses)
        self.cooldowns.update(
            {
                "restart_service": 300,  # 5 minutes between restarts
                "install_package": 1800,  # 30 minutes between installs
                "make_payment": 3600,  # 1 hour between payments
                "delete_user": 3600,  # 1 hour between user deletions
            }
        )

    # Policy management methods
    def add_rate_limit(self, tool_name: str, max_calls: int, per_seconds: int):
        """Add a rate limit for a specific tool."""
        self.rate_limits[tool_name] = (max_calls, per_seconds)

    def add_confirmation_required(self, tool_name: str):
        """Add a tool to the confirmation required list."""
        self.confirmation_required.add(tool_name)

    def add_admin_required(self, tool_name: str):
        """Add a tool to the admin required list."""
        self.admin_required.add(tool_name)

    def add_cooldown(self, tool_name: str, cooldown_seconds: int):
        """Add a cooldown period for a specific tool."""
        self.cooldowns[tool_name] = cooldown_seconds

    def remove_rate_limit(self, tool_name: str):
        """Remove rate limit for a specific tool."""
        self.rate_limits.pop(tool_name, None)

    def remove_confirmation_required(self, tool_name: str):
        """Remove a tool from the confirmation required list."""
        self.confirmation_required.discard(tool_name)

    def remove_admin_required(self, tool_name: str):
        """Remove a tool from the admin required list."""
        self.admin_required.discard(tool_name)

    def remove_cooldown(self, tool_name: str):
        """Remove cooldown for a specific tool."""
        self.cooldowns.pop(tool_name, None)

    def clear_all_guards(self):
        """Remove all guard policies."""
        self.rate_limits.clear()
        self.confirmation_required.clear()
        self.admin_required.clear()
        self.cooldowns.clear()

    # Policy checking methods
    def can_use(self, tool_name: str) -> bool:
        """Check if a tool can be used based on rate limits and cooldowns."""
        now = time.time()

        # Check rate limits
        if tool_name in self.rate_limits:
            max_calls, per_seconds = self.rate_limits[tool_name]
            timestamps = self.usage_log.get(tool_name, [])
            # Only keep timestamps within the window
            timestamps = [t for t in timestamps if now - t < per_seconds]
            if len(timestamps) >= max_calls:
                return False
            self.usage_log[tool_name] = timestamps

        # Check cooldowns
        if tool_name in self.cooldowns:
            cooldown_seconds = self.cooldowns[tool_name]
            timestamps = self.usage_log.get(tool_name, [])
            if timestamps and (now - timestamps[-1]) < cooldown_seconds:
                return False

        return True

    def record_use(self, tool_name: str):
        """Record the usage of a tool."""
        self.usage_log.setdefault(tool_name, []).append(time.time())

    def needs_confirmation(self, tool_name: str) -> bool:
        """Check if a tool requires confirmation."""
        return tool_name in self.confirmation_required

    def needs_admin(self, tool_name: str) -> bool:
        """Check if a tool requires admin privileges."""
        return tool_name in self.admin_required

    # Information methods
    def get_rate_limit_info(self, tool_name: str) -> Dict[str, float]:
        """Get information about rate limits for a tool."""
        if tool_name in self.rate_limits:
            max_calls, per_seconds = self.rate_limits[tool_name]
            timestamps = self.usage_log.get(tool_name, [])
            now = time.time()
            recent_calls = len([t for t in timestamps if now - t < per_seconds])
            return {
                "max_calls": max_calls,
                "per_seconds": per_seconds,
                "recent_calls": recent_calls,
                "remaining_calls": max_calls - recent_calls,
                "window_remaining": (
                    per_seconds - (now - timestamps[-1]) if timestamps else 0
                ),
            }
        return {}

    def get_cooldown_info(self, tool_name: str) -> Dict[str, float]:
        """Get information about cooldown for a tool."""
        if tool_name in self.cooldowns:
            cooldown_seconds = self.cooldowns[tool_name]
            timestamps = self.usage_log.get(tool_name, [])
            now = time.time()
            if timestamps:
                time_since_last = now - timestamps[-1]
                cooldown_remaining = max(0, cooldown_seconds - time_since_last)
                return {
                    "cooldown_seconds": cooldown_seconds,
                    "time_since_last": time_since_last,
                    "cooldown_remaining": cooldown_remaining,
                    "can_use": cooldown_remaining <= 0,
                }
        return {}