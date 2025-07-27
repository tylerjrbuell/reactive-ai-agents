"""
Error Recovery Orchestrator for centralized error handling with learning capabilities.

This module provides intelligent error recovery with pattern matching and
learning capabilities to improve recovery strategies over time.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import json
from collections import defaultdict
from pydantic import BaseModel, Field

from reactive_agents.core.reasoning.protocols import ComponentResult, RetryStrategy
from reactive_agents.core.types.reasoning_component_types import ErrorRecoveryResult


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors for pattern matching."""
    NETWORK = "network"
    VALIDATION = "validation"
    TIMEOUT = "timeout"
    RESOURCE = "resource"
    LOGIC = "logic"
    EXTERNAL_SERVICE = "external_service"
    CONFIGURATION = "configuration"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    UNKNOWN = "unknown"


class RecoveryAction(Enum):
    """Available recovery actions."""
    RETRY = "retry"
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    SWITCH_STRATEGY = "switch_strategy"
    SIMPLIFY_APPROACH = "simplify_approach"
    REQUEST_USER_INPUT = "request_user_input"
    FALLBACK_MODE = "fallback_mode"
    TERMINATE = "terminate"
    ESCALATE = "escalate"
    RESET_CONTEXT = "reset_context"
    REDUCE_SCOPE = "reduce_scope"


@dataclass
class ErrorContext:
    """Context information for an error occurrence."""
    task: str
    component: str
    operation: str
    iteration: int
    strategy: str
    error_message: str
    error_type: str
    stack_trace: Optional[str] = None
    context_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RecoveryPattern:
    """A learned pattern for error recovery."""
    error_signature: str  # Hash of error characteristics
    error_category: ErrorCategory
    severity: ErrorSeverity
    recovery_action: RecoveryAction
    success_rate: float = 0.0
    usage_count: int = 0
    last_used: Optional[datetime] = None
    context_requirements: Set[str] = field(default_factory=set)
    effectiveness_score: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FailureRecord:
    """Record of a failure and recovery attempt."""
    error_context: ErrorContext
    recovery_pattern: Optional[RecoveryPattern]
    recovery_action_taken: RecoveryAction
    success: bool
    recovery_time_ms: float
    timestamp: datetime = field(default_factory=datetime.now)
    notes: str = ""


class RecoveryResult(BaseModel):
    """Result of an error recovery operation."""
    success: bool
    recovery_action: RecoveryAction
    rationale: str
    alternative_strategy: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    estimated_recovery_time_ms: Optional[float] = None
    requires_user_input: bool = False
    updated_context: Dict[str, Any] = Field(default_factory=dict)
    should_retry: bool = False
    should_switch_strategy: bool = False
    recommended_strategy: Optional[str] = None
    prevention_recommendations: List[str] = Field(default_factory=list)


class ErrorClassifier:
    """Classifies errors into categories and determines severity."""

    def __init__(self):
        self.category_patterns = {
            ErrorCategory.NETWORK: [
                'connection', 'timeout', 'dns', 'socket', 'http', 'ssl', 'tls'
            ],
            ErrorCategory.VALIDATION: [
                'validation', 'schema', 'format', 'parse', 'invalid', 'malformed'
            ],
            ErrorCategory.TIMEOUT: [
                'timeout', 'deadline', 'expired', 'took too long'
            ],
            ErrorCategory.RESOURCE: [
                'memory', 'disk', 'cpu', 'quota', 'limit', 'resource'
            ],
            ErrorCategory.AUTHENTICATION: [
                'auth', 'permission', 'credential', 'unauthorized', 'forbidden'
            ],
            ErrorCategory.RATE_LIMIT: [
                'rate limit', 'throttle', 'too many requests', 'quota exceeded'
            ],
            ErrorCategory.EXTERNAL_SERVICE: [
                'service unavailable', 'external', 'api', 'downstream'
            ],
        }

        self.severity_indicators = {
            ErrorSeverity.CRITICAL: [
                'critical', 'fatal', 'severe', 'emergency', 'panic'
            ],
            ErrorSeverity.HIGH: [
                'error', 'failed', 'exception', 'crash', 'abort'
            ],
            ErrorSeverity.MEDIUM: [
                'warning', 'issue', 'problem', 'unexpected'
            ],
            ErrorSeverity.LOW: [
                'notice', 'info', 'minor', 'trivial'
            ],
        }

    def classify_error(self, error_context: ErrorContext) -> Tuple[ErrorCategory, ErrorSeverity]:
        """
        Classify an error into category and severity.
        
        Args:
            error_context: The error context to classify
            
        Returns:
            Tuple of (category, severity)
        """
        error_text = (
            error_context.error_message + " " + 
            error_context.error_type + " " +
            (error_context.stack_trace or "")
        ).lower()

        # Determine category
        category = ErrorCategory.UNKNOWN
        for cat, patterns in self.category_patterns.items():
            if any(pattern in error_text for pattern in patterns):
                category = cat
                break

        # Determine severity
        severity = ErrorSeverity.MEDIUM  # Default
        for sev, indicators in self.severity_indicators.items():
            if any(indicator in error_text for indicator in indicators):
                severity = sev
                break

        return category, severity

    def generate_error_signature(self, error_context: ErrorContext) -> str:
        """Generate a unique signature for an error pattern."""
        signature_data = {
            'error_type': error_context.error_type,
            'component': error_context.component,
            'operation': error_context.operation,
            'message_pattern': self._extract_message_pattern(error_context.error_message)
        }
        
        signature_str = json.dumps(signature_data, sort_keys=True)
        return hashlib.sha256(signature_str.encode()).hexdigest()[:16]

    def _extract_message_pattern(self, error_message: str) -> str:
        """Extract a generalized pattern from an error message."""
        # Remove specific values like IDs, paths, numbers
        import re
        pattern = re.sub(r'\d+', '[NUM]', error_message)
        pattern = re.sub(r'/[^\s]+', '[PATH]', pattern)
        pattern = re.sub(r'[a-f0-9]{8,}', '[ID]', pattern)
        return pattern.lower()


class ErrorRecoveryOrchestrator:
    """
    Centralized error recovery system with learning capabilities.
    
    This orchestrator manages error recovery strategies, learns from
    past recovery attempts, and adapts its approach over time.
    """

    def __init__(self, max_pattern_history: int = 1000):
        """
        Initialize the error recovery orchestrator.
        
        Args:
            max_pattern_history: Maximum number of patterns to keep in memory
        """
        self.classifier = ErrorClassifier()
        self.recovery_patterns: Dict[str, RecoveryPattern] = {}
        self.failure_history: List[FailureRecord] = []
        self.max_pattern_history = max_pattern_history
        
        # Strategy effectiveness tracking
        self.strategy_performance: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        # Custom recovery handlers
        self.custom_handlers: Dict[ErrorCategory, Callable] = {}
        
        # Recovery statistics
        self.recovery_stats = {
            'total_recoveries': 0,
            'successful_recoveries': 0,
            'pattern_matches': 0,
            'new_patterns_learned': 0
        }

        # Initialize default recovery patterns
        self._initialize_default_patterns()

    def _initialize_default_patterns(self) -> None:
        """Initialize default recovery patterns for common error types."""
        default_patterns = [
            RecoveryPattern(
                error_signature="network_timeout",
                error_category=ErrorCategory.NETWORK,
                severity=ErrorSeverity.MEDIUM,
                recovery_action=RecoveryAction.RETRY_WITH_BACKOFF,
                success_rate=0.7,
                effectiveness_score=0.8
            ),
            RecoveryPattern(
                error_signature="validation_error",
                error_category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.HIGH,
                recovery_action=RecoveryAction.SIMPLIFY_APPROACH,
                success_rate=0.6,
                effectiveness_score=0.7
            ),
            RecoveryPattern(
                error_signature="rate_limit",
                error_category=ErrorCategory.RATE_LIMIT,
                severity=ErrorSeverity.MEDIUM,
                recovery_action=RecoveryAction.RETRY_WITH_BACKOFF,
                success_rate=0.9,
                effectiveness_score=0.9
            ),
            RecoveryPattern(
                error_signature="resource_exhausted",
                error_category=ErrorCategory.RESOURCE,
                severity=ErrorSeverity.HIGH,
                recovery_action=RecoveryAction.REDUCE_SCOPE,
                success_rate=0.5,
                effectiveness_score=0.6
            ),
        ]
        
        for pattern in default_patterns:
            self.recovery_patterns[pattern.error_signature] = pattern

    async def handle_error(
        self, 
        error: Exception, 
        error_context: ErrorContext
    ) -> RecoveryResult:
        """
        Handle an error with intelligent recovery strategies.
        
        Args:
            error: The exception that occurred
            error_context: Context information about the error
            
        Returns:
            RecoveryResult with recommended actions
        """
        start_time = datetime.now()
        
        # Update error context with exception details
        error_context.error_type = type(error).__name__
        if not error_context.error_message:
            error_context.error_message = str(error)

        # Classify the error
        category, severity = self.classifier.classify_error(error_context)
        error_signature = self.classifier.generate_error_signature(error_context)

        # Look for existing recovery pattern
        pattern = self._find_recovery_pattern(error_signature, category, error_context)

        # Generate recovery strategy
        if pattern and pattern.success_rate > 0.3:  # Use existing pattern if it's reasonably successful
            recovery_result = await self._apply_existing_pattern(pattern, error_context)
        else:
            recovery_result = await self._generate_new_recovery_strategy(
                category, severity, error_context
            )

        # Record the recovery attempt
        recovery_time = (datetime.now() - start_time).total_seconds() * 1000
        
        failure_record = FailureRecord(
            error_context=error_context,
            recovery_pattern=pattern,
            recovery_action_taken=recovery_result.recovery_action,
            success=False,  # Will be updated when we know the outcome
            recovery_time_ms=recovery_time
        )
        
        self.failure_history.append(failure_record)
        self.recovery_stats['total_recoveries'] += 1
        
        if pattern:
            self.recovery_stats['pattern_matches'] += 1

        return recovery_result

    def _find_recovery_pattern(
        self, 
        error_signature: str, 
        category: ErrorCategory,
        error_context: ErrorContext
    ) -> Optional[RecoveryPattern]:
        """Find the best matching recovery pattern."""
        # First, try exact signature match
        if error_signature in self.recovery_patterns:
            pattern = self.recovery_patterns[error_signature]
            pattern.usage_count += 1
            pattern.last_used = datetime.now()
            return pattern

        # Then, try category-based matching
        category_patterns = [
            p for p in self.recovery_patterns.values()
            if p.error_category == category and p.success_rate > 0.4
        ]
        
        if category_patterns:
            # Return the pattern with highest effectiveness score
            return max(category_patterns, key=lambda p: p.effectiveness_score)

        return None

    async def _apply_existing_pattern(
        self, 
        pattern: RecoveryPattern, 
        error_context: ErrorContext
    ) -> RecoveryResult:
        """Apply an existing recovery pattern."""
        confidence = min(0.9, pattern.success_rate + 0.1)
        
        return RecoveryResult(
            success=True,
            recovery_action=pattern.recovery_action,
            rationale=f"Using learned pattern (success rate: {pattern.success_rate:.1%})",
            confidence=confidence,
            should_retry=pattern.recovery_action in [
                RecoveryAction.RETRY, 
                RecoveryAction.RETRY_WITH_BACKOFF
            ],
            should_switch_strategy=pattern.recovery_action == RecoveryAction.SWITCH_STRATEGY,
            prevention_recommendations=self._generate_prevention_recommendations(
                pattern.error_category
            )
        )

    async def _generate_new_recovery_strategy(
        self,
        category: ErrorCategory,
        severity: ErrorSeverity,
        error_context: ErrorContext
    ) -> RecoveryResult:
        """Generate a new recovery strategy for an unknown error pattern."""
        
        # Use heuristics based on category and severity
        recovery_action = self._determine_recovery_action(category, severity)
        confidence = self._calculate_base_confidence(category, severity)
        
        # Create new pattern
        new_pattern = RecoveryPattern(
            error_signature=self.classifier.generate_error_signature(error_context),
            error_category=category,
            severity=severity,
            recovery_action=recovery_action,
            success_rate=0.5,  # Start with neutral assumption
            usage_count=1,
            last_used=datetime.now(),
            effectiveness_score=confidence
        )
        
        self.recovery_patterns[new_pattern.error_signature] = new_pattern
        self.recovery_stats['new_patterns_learned'] += 1
        
        return RecoveryResult(
            success=True,
            recovery_action=recovery_action,
            rationale=f"New {category.value} error - applying heuristic recovery",
            confidence=confidence,
            should_retry=recovery_action in [
                RecoveryAction.RETRY, 
                RecoveryAction.RETRY_WITH_BACKOFF
            ],
            should_switch_strategy=recovery_action == RecoveryAction.SWITCH_STRATEGY,
            prevention_recommendations=self._generate_prevention_recommendations(category)
        )

    def _determine_recovery_action(
        self, 
        category: ErrorCategory, 
        severity: ErrorSeverity
    ) -> RecoveryAction:
        """Determine the appropriate recovery action based on category and severity."""
        
        # Critical errors generally require termination or escalation
        if severity == ErrorSeverity.CRITICAL:
            return RecoveryAction.TERMINATE
        
        # Category-specific logic
        action_map = {
            ErrorCategory.NETWORK: RecoveryAction.RETRY_WITH_BACKOFF,
            ErrorCategory.TIMEOUT: RecoveryAction.RETRY_WITH_BACKOFF,
            ErrorCategory.RATE_LIMIT: RecoveryAction.RETRY_WITH_BACKOFF,
            ErrorCategory.VALIDATION: RecoveryAction.SIMPLIFY_APPROACH,
            ErrorCategory.RESOURCE: RecoveryAction.REDUCE_SCOPE,
            ErrorCategory.AUTHENTICATION: RecoveryAction.ESCALATE,
            ErrorCategory.EXTERNAL_SERVICE: RecoveryAction.FALLBACK_MODE,
            ErrorCategory.CONFIGURATION: RecoveryAction.RESET_CONTEXT,
            ErrorCategory.LOGIC: RecoveryAction.SWITCH_STRATEGY,
        }
        
        return action_map.get(category, RecoveryAction.RETRY)

    def _calculate_base_confidence(
        self, 
        category: ErrorCategory, 
        severity: ErrorSeverity
    ) -> float:
        """Calculate base confidence for a recovery strategy."""
        # Start with category-based confidence
        category_confidence = {
            ErrorCategory.NETWORK: 0.7,  # Network errors often recoverable
            ErrorCategory.TIMEOUT: 0.8,  # Timeouts usually recoverable
            ErrorCategory.RATE_LIMIT: 0.9,  # Rate limits very recoverable
            ErrorCategory.VALIDATION: 0.6,  # Validation errors moderately recoverable
            ErrorCategory.RESOURCE: 0.5,  # Resource issues harder to recover
            ErrorCategory.AUTHENTICATION: 0.3,  # Auth issues often require intervention
            ErrorCategory.EXTERNAL_SERVICE: 0.6,  # External service issues vary
            ErrorCategory.CONFIGURATION: 0.4,  # Config issues often require fixes
            ErrorCategory.LOGIC: 0.5,  # Logic errors require different approach
        }.get(category, 0.4)
        
        # Adjust based on severity
        severity_modifier = {
            ErrorSeverity.LOW: 1.1,
            ErrorSeverity.MEDIUM: 1.0,
            ErrorSeverity.HIGH: 0.8,
            ErrorSeverity.CRITICAL: 0.3,
        }.get(severity, 1.0)
        
        return min(0.9, category_confidence * severity_modifier)

    def _generate_prevention_recommendations(
        self, 
        category: ErrorCategory
    ) -> List[str]:
        """Generate prevention recommendations based on error category."""
        recommendations = {
            ErrorCategory.NETWORK: [
                "Implement circuit breaker patterns",
                "Add connection pooling",
                "Use exponential backoff for retries"
            ],
            ErrorCategory.VALIDATION: [
                "Add input validation earlier in the pipeline",
                "Implement schema validation",
                "Add data sanitization steps"
            ],
            ErrorCategory.TIMEOUT: [
                "Increase timeout values for complex operations",
                "Implement streaming for large data",
                "Add progress monitoring"
            ],
            ErrorCategory.RESOURCE: [
                "Monitor resource usage",
                "Implement resource cleanup",
                "Add memory management optimizations"
            ],
            ErrorCategory.RATE_LIMIT: [
                "Implement rate limiting awareness",
                "Add request queuing",
                "Use exponential backoff"
            ]
        }
        
        return recommendations.get(category, ["Monitor error patterns", "Add logging"])

    def learn_from_recovery(self, recovery_attempt: FailureRecord, success: bool) -> None:
        """
        Learn from a recovery attempt to improve future strategies.
        
        Args:
            recovery_attempt: The recovery attempt record
            success: Whether the recovery was successful
        """
        # Update the failure record
        recovery_attempt.success = success
        
        if success:
            self.recovery_stats['successful_recoveries'] += 1

        # Update pattern effectiveness if we have one
        if recovery_attempt.recovery_pattern:
            pattern = recovery_attempt.recovery_pattern
            
            # Update success rate using exponential moving average
            alpha = 0.1  # Learning rate
            if success:
                pattern.success_rate = (1 - alpha) * pattern.success_rate + alpha * 1.0
            else:
                pattern.success_rate = (1 - alpha) * pattern.success_rate + alpha * 0.0
            
            # Update effectiveness score
            if success:
                pattern.effectiveness_score = min(0.95, pattern.effectiveness_score + 0.05)
            else:
                pattern.effectiveness_score = max(0.1, pattern.effectiveness_score - 0.1)

        # Clean up old patterns if we're over the limit
        if len(self.recovery_patterns) > self.max_pattern_history:
            self._cleanup_old_patterns()

    def _cleanup_old_patterns(self) -> None:
        """Remove old or ineffective patterns to keep memory usage reasonable."""
        # Sort patterns by effectiveness and age
        patterns_list = list(self.recovery_patterns.items())
        
        # Remove patterns that haven't been used in 30 days and have low effectiveness
        cutoff_date = datetime.now() - timedelta(days=30)
        
        patterns_to_remove = [
            sig for sig, pattern in patterns_list
            if (pattern.last_used and pattern.last_used < cutoff_date and 
                pattern.effectiveness_score < 0.3)
        ]
        
        for sig in patterns_to_remove:
            del self.recovery_patterns[sig]

    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get comprehensive recovery statistics."""
        if self.recovery_stats['total_recoveries'] > 0:
            success_rate = (
                self.recovery_stats['successful_recoveries'] / 
                self.recovery_stats['total_recoveries']
            )
        else:
            success_rate = 0.0
        
        return {
            **self.recovery_stats,
            'overall_success_rate': success_rate,
            'active_patterns': len(self.recovery_patterns),
            'pattern_effectiveness': {
                sig: {
                    'success_rate': pattern.success_rate,
                    'usage_count': pattern.usage_count,
                    'effectiveness_score': pattern.effectiveness_score
                }
                for sig, pattern in self.recovery_patterns.items()
            },
            'category_distribution': self._get_category_distribution()
        }

    def _get_category_distribution(self) -> Dict[str, int]:
        """Get distribution of errors by category."""
        distribution = defaultdict(int)
        for record in self.failure_history[-100:]:  # Last 100 records
            category = self.classifier.classify_error(record.error_context)[0]
            distribution[category.value] += 1
        return dict(distribution)