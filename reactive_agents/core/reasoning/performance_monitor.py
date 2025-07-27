"""
Strategy Performance Monitor for real-time strategy performance tracking.

This module provides comprehensive performance monitoring and analysis
for reasoning strategies, enabling data-driven strategy selection and optimization.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import statistics
from collections import defaultdict, deque
from pydantic import BaseModel, Field

from reactive_agents.core.reasoning.state_machine import StrategyState


class PerformanceMetric(Enum):
    """Available performance metrics."""
    SUCCESS_RATE = "success_rate"
    AVERAGE_ITERATIONS = "average_iterations"
    EXECUTION_TIME = "execution_time"
    ERROR_RATE = "error_rate"
    COMPLETION_SCORE = "completion_score"
    EFFICIENCY_SCORE = "efficiency_score"
    RESOURCE_USAGE = "resource_usage"
    CONTEXT_EFFICIENCY = "context_efficiency"
    TOOL_USAGE_EFFECTIVENESS = "tool_usage_effectiveness"


class PerformanceThreshold(BaseModel):
    """Performance thresholds for strategy evaluation."""
    min_success_rate: float = Field(default=0.7, ge=0.0, le=1.0)
    max_average_iterations: int = Field(default=10, ge=1)
    max_execution_time_ms: float = Field(default=300000, ge=0)  # 5 minutes
    max_error_rate: float = Field(default=0.3, ge=0.0, le=1.0)
    min_completion_score: float = Field(default=0.6, ge=0.0, le=1.0)
    min_efficiency_score: float = Field(default=0.5, ge=0.0, le=1.0)


@dataclass
class ExecutionRecord:
    """Record of a single strategy execution."""
    strategy_name: str
    task_description: str
    start_time: datetime
    end_time: Optional[datetime] = None
    success: bool = False
    iterations: int = 0
    completion_score: float = 0.0
    error_count: int = 0
    state_transitions: List[Tuple[StrategyState, StrategyState]] = field(default_factory=list)
    tool_calls: int = 0
    context_tokens_used: int = 0
    resource_metrics: Dict[str, float] = field(default_factory=dict)
    error_messages: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def execution_time_ms(self) -> float:
        """Get execution time in milliseconds."""
        if not self.end_time:
            return 0.0
        return (self.end_time - self.start_time).total_seconds() * 1000

    @property
    def efficiency_score(self) -> float:
        """Calculate efficiency score based on completion vs. resources used."""
        if self.iterations == 0:
            return 0.0
        
        # Base efficiency on completion score per iteration
        base_efficiency = self.completion_score / max(1, self.iterations)
        
        # Adjust for error rate
        error_penalty = self.error_count * 0.1
        
        return max(0.0, min(1.0, base_efficiency - error_penalty))


class StrategyMetrics(BaseModel):
    """Aggregated metrics for a strategy."""
    strategy_name: str
    total_executions: int = 0
    successful_executions: int = 0
    total_iterations: int = 0
    total_execution_time_ms: float = 0.0
    total_errors: int = 0
    completion_scores: List[float] = Field(default_factory=list)
    efficiency_scores: List[float] = Field(default_factory=list)
    recent_performance_window: deque = Field(default_factory=lambda: deque(maxlen=20))
    
    class Config:
        arbitrary_types_allowed = True

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_executions == 0:
            return 0.0
        return self.successful_executions / self.total_executions

    @property
    def average_iterations(self) -> float:
        """Calculate average iterations per execution."""
        if self.total_executions == 0:
            return 0.0
        return self.total_iterations / self.total_executions

    @property
    def average_execution_time_ms(self) -> float:
        """Calculate average execution time."""
        if self.total_executions == 0:
            return 0.0
        return self.total_execution_time_ms / self.total_executions

    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        if self.total_executions == 0:
            return 0.0
        return self.total_errors / max(1, self.total_iterations)

    @property
    def average_completion_score(self) -> float:
        """Calculate average completion score."""
        if not self.completion_scores:
            return 0.0
        return statistics.mean(self.completion_scores)

    @property
    def average_efficiency_score(self) -> float:
        """Calculate average efficiency score."""
        if not self.efficiency_scores:
            return 0.0
        return statistics.mean(self.efficiency_scores)

    @property
    def recent_success_rate(self) -> float:
        """Calculate success rate for recent executions."""
        if not self.recent_performance_window:
            return self.success_rate
        
        recent_successes = sum(1 for record in self.recent_performance_window if record['success'])
        return recent_successes / len(self.recent_performance_window)

    def calculate_overall_score(self) -> float:
        """Calculate an overall performance score."""
        weights = {
            'success_rate': 0.3,
            'efficiency': 0.25,
            'completion': 0.25,
            'speed': 0.1,
            'reliability': 0.1
        }
        
        # Normalize speed metric (lower is better)
        speed_score = max(0, 1 - (self.average_execution_time_ms / 300000))  # 5 min baseline
        
        # Reliability based on error rate (lower is better)
        reliability_score = max(0, 1 - self.error_rate)
        
        overall_score = (
            weights['success_rate'] * self.success_rate +
            weights['efficiency'] * self.average_efficiency_score +
            weights['completion'] * self.average_completion_score +
            weights['speed'] * speed_score +
            weights['reliability'] * reliability_score
        )
        
        return min(1.0, max(0.0, overall_score))


class PerformanceReport(BaseModel):
    """Comprehensive performance report for a strategy."""
    strategy_name: str
    metrics: StrategyMetrics
    efficiency_score: float
    success_rate: float
    average_iterations: float
    error_patterns: List[str]
    recommendation: str
    confidence: float = Field(ge=0.0, le=1.0)
    trend_analysis: Dict[str, str] = Field(default_factory=dict)
    comparative_ranking: Optional[int] = None
    improvement_suggestions: List[str] = Field(default_factory=list)
    
    @classmethod
    def empty(cls, strategy_name: str = "unknown") -> PerformanceReport:
        """Create an empty performance report."""
        return cls(
            strategy_name=strategy_name,
            metrics=StrategyMetrics(strategy_name=strategy_name),
            efficiency_score=0.0,
            success_rate=0.0,
            average_iterations=0.0,
            error_patterns=[],
            recommendation="Insufficient data for analysis",
            confidence=0.0
        )


class StrategyPerformanceMonitor:
    """
    Real-time strategy performance tracking and analysis system.
    
    This monitor tracks execution metrics, identifies performance patterns,
    and provides intelligent recommendations for strategy selection and optimization.
    """

    def __init__(self, history_retention_days: int = 30):
        """
        Initialize the performance monitor.
        
        Args:
            history_retention_days: Number of days to retain execution history
        """
        self.history_retention_days = history_retention_days
        self.execution_history: List[ExecutionRecord] = []
        self.metrics_cache: Dict[str, StrategyMetrics] = {}
        self.thresholds = PerformanceThreshold()
        self.active_executions: Dict[str, ExecutionRecord] = {}
        
        # Performance tracking
        self.performance_trends: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self.comparative_rankings: Dict[str, int] = {}
        
        # Analysis configuration
        self.analysis_window_days = 7
        self.min_executions_for_analysis = 3

    def start_execution_tracking(
        self, 
        execution_id: str,
        strategy_name: str, 
        task_description: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Start tracking a new strategy execution.
        
        Args:
            execution_id: Unique identifier for this execution
            strategy_name: Name of the strategy being executed
            task_description: Description of the task
            metadata: Additional metadata for this execution
        """
        record = ExecutionRecord(
            strategy_name=strategy_name,
            task_description=task_description,
            start_time=datetime.now(),
            metadata=metadata or {}
        )
        
        self.active_executions[execution_id] = record

    def update_execution_progress(
        self,
        execution_id: str,
        iterations: Optional[int] = None,
        error_count: Optional[int] = None,
        tool_calls: Optional[int] = None,
        context_tokens: Optional[int] = None,
        state_transition: Optional[Tuple[StrategyState, StrategyState]] = None,
        error_message: Optional[str] = None
    ) -> None:
        """
        Update progress for an active execution.
        
        Args:
            execution_id: Execution identifier
            iterations: Current iteration count
            error_count: Current error count
            tool_calls: Number of tool calls made
            context_tokens: Number of context tokens used
            state_transition: State transition that occurred
            error_message: Error message if an error occurred
        """
        if execution_id not in self.active_executions:
            return
        
        record = self.active_executions[execution_id]
        
        if iterations is not None:
            record.iterations = iterations
        if error_count is not None:
            record.error_count = error_count
        if tool_calls is not None:
            record.tool_calls = tool_calls
        if context_tokens is not None:
            record.context_tokens_used = context_tokens
        if state_transition:
            record.state_transitions.append(state_transition)
        if error_message:
            record.error_messages.append(error_message)

    def complete_execution_tracking(
        self,
        execution_id: str,
        success: bool,
        completion_score: float,
        final_metadata: Optional[Dict[str, Any]] = None
    ) -> ExecutionRecord:
        """
        Complete tracking for an execution.
        
        Args:
            execution_id: Execution identifier
            success: Whether the execution was successful
            completion_score: Final completion score (0.0 to 1.0)
            final_metadata: Final metadata to add
            
        Returns:
            The completed execution record
        """
        if execution_id not in self.active_executions:
            raise ValueError(f"No active execution found with ID: {execution_id}")
        
        record = self.active_executions.pop(execution_id)
        record.end_time = datetime.now()
        record.success = success
        record.completion_score = completion_score
        
        if final_metadata:
            record.metadata.update(final_metadata)
        
        # Add to history
        self.execution_history.append(record)
        
        # Update metrics cache
        self._update_metrics_cache(record)
        
        # Update performance trends
        self._update_performance_trends(record)
        
        # Clean up old history
        self._cleanup_old_history()
        
        return record

    def _update_metrics_cache(self, record: ExecutionRecord) -> None:
        """Update the metrics cache with a new execution record."""
        strategy_name = record.strategy_name
        
        if strategy_name not in self.metrics_cache:
            self.metrics_cache[strategy_name] = StrategyMetrics(strategy_name=strategy_name)
        
        metrics = self.metrics_cache[strategy_name]
        
        # Update counters
        metrics.total_executions += 1
        if record.success:
            metrics.successful_executions += 1
        
        metrics.total_iterations += record.iterations
        metrics.total_execution_time_ms += record.execution_time_ms
        metrics.total_errors += record.error_count
        
        # Update score lists
        metrics.completion_scores.append(record.completion_score)
        metrics.efficiency_scores.append(record.efficiency_score)
        
        # Update recent performance window
        metrics.recent_performance_window.append({
            'success': record.success,
            'completion_score': record.completion_score,
            'efficiency_score': record.efficiency_score,
            'execution_time_ms': record.execution_time_ms
        })

    def _update_performance_trends(self, record: ExecutionRecord) -> None:
        """Update performance trend tracking."""
        strategy_name = record.strategy_name
        
        trend_data = {
            'timestamp': record.end_time.isoformat(),
            'success': record.success,
            'completion_score': record.completion_score,
            'efficiency_score': record.efficiency_score,
            'execution_time_ms': record.execution_time_ms,
            'iterations': record.iterations
        }
        
        self.performance_trends[strategy_name].append(trend_data)

    def _cleanup_old_history(self) -> None:
        """Clean up old execution history."""
        cutoff_date = datetime.now() - timedelta(days=self.history_retention_days)
        
        self.execution_history = [
            record for record in self.execution_history
            if record.start_time > cutoff_date
        ]
        
        # Rebuild metrics cache after cleanup
        self._rebuild_metrics_cache()

    def _rebuild_metrics_cache(self) -> None:
        """Rebuild the metrics cache from current history."""
        self.metrics_cache.clear()
        
        for record in self.execution_history:
            self._update_metrics_cache(record)

    async def evaluate_strategy_performance(self, strategy_name: str) -> PerformanceReport:
        """
        Generate a comprehensive performance report for a strategy.
        
        Args:
            strategy_name: Name of the strategy to evaluate
            
        Returns:
            Detailed performance report
        """
        if strategy_name not in self.metrics_cache:
            return PerformanceReport.empty(strategy_name)
        
        metrics = self.metrics_cache[strategy_name]
        
        # Check if we have enough data for meaningful analysis
        if metrics.total_executions < self.min_executions_for_analysis:
            return PerformanceReport(
                strategy_name=strategy_name,
                metrics=metrics,
                efficiency_score=metrics.average_efficiency_score,
                success_rate=metrics.success_rate,
                average_iterations=metrics.average_iterations,
                error_patterns=self._analyze_error_patterns(strategy_name),
                recommendation="Insufficient execution data for reliable analysis",
                confidence=0.3,
                improvement_suggestions=["Increase sample size by running more executions"]
            )
        
        # Perform trend analysis
        trend_analysis = self._analyze_performance_trends(strategy_name)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(metrics)
        
        # Calculate confidence based on data quality
        confidence = self._calculate_confidence(metrics)
        
        # Generate improvement suggestions
        improvement_suggestions = self._generate_improvement_suggestions(metrics)
        
        return PerformanceReport(
            strategy_name=strategy_name,
            metrics=metrics,
            efficiency_score=metrics.average_efficiency_score,
            success_rate=metrics.success_rate,
            average_iterations=metrics.average_iterations,
            error_patterns=self._analyze_error_patterns(strategy_name),
            recommendation=recommendation,
            confidence=confidence,
            trend_analysis=trend_analysis,
            comparative_ranking=self.comparative_rankings.get(strategy_name),
            improvement_suggestions=improvement_suggestions
        )

    def should_switch_strategy(self, current_strategy: str) -> Optional[str]:
        """
        Determine if the current strategy should be switched.
        
        Args:
            current_strategy: Name of the current strategy
            
        Returns:
            Recommended alternative strategy name, or None if no switch needed
        """
        if current_strategy not in self.metrics_cache:
            return None
        
        current_metrics = self.metrics_cache[current_strategy]
        
        # Check if current strategy is underperforming
        if (current_metrics.recent_success_rate < self.thresholds.min_success_rate or
            current_metrics.average_efficiency_score < self.thresholds.min_efficiency_score):
            
            # Find better performing alternative
            return self._find_best_alternative_strategy(current_strategy)
        
        return None

    def _find_best_alternative_strategy(self, current_strategy: str) -> Optional[str]:
        """Find the best performing alternative strategy."""
        alternatives = [
            (name, metrics) for name, metrics in self.metrics_cache.items()
            if (name != current_strategy and 
                metrics.total_executions >= self.min_executions_for_analysis)
        ]
        
        if not alternatives:
            return None
        
        # Sort by overall performance score
        alternatives.sort(key=lambda x: x[1].calculate_overall_score(), reverse=True)
        
        best_alternative = alternatives[0]
        
        # Only recommend if significantly better
        current_score = self.metrics_cache[current_strategy].calculate_overall_score()
        if best_alternative[1].calculate_overall_score() > current_score + 0.1:
            return best_alternative[0]
        
        return None

    def _analyze_error_patterns(self, strategy_name: str) -> List[str]:
        """Analyze error patterns for a strategy."""
        strategy_records = [
            record for record in self.execution_history
            if record.strategy_name == strategy_name
        ]
        
        error_messages = []
        for record in strategy_records:
            error_messages.extend(record.error_messages)
        
        # Simple pattern analysis (could be enhanced with NLP)
        error_patterns = []
        error_counts = defaultdict(int)
        
        for error_msg in error_messages:
            # Extract common error patterns
            if "timeout" in error_msg.lower():
                error_counts["timeout_errors"] += 1
            elif "validation" in error_msg.lower():
                error_counts["validation_errors"] += 1
            elif "connection" in error_msg.lower():
                error_counts["connection_errors"] += 1
            elif "permission" in error_msg.lower():
                error_counts["permission_errors"] += 1
            else:
                error_counts["other_errors"] += 1
        
        for error_type, count in error_counts.items():
            if count > 0:
                error_patterns.append(f"{error_type}: {count} occurrences")
        
        return error_patterns

    def _analyze_performance_trends(self, strategy_name: str) -> Dict[str, str]:
        """Analyze performance trends for a strategy."""
        if strategy_name not in self.performance_trends:
            return {}
        
        trends = list(self.performance_trends[strategy_name])
        if len(trends) < 5:
            return {"trend": "insufficient_data"}
        
        # Analyze recent vs. older performance
        recent_trends = trends[-10:]  # Last 10 executions
        older_trends = trends[-20:-10] if len(trends) >= 20 else trends[:-10]
        
        trend_analysis = {}
        
        # Success rate trend
        if older_trends:
            recent_success = sum(1 for t in recent_trends if t['success']) / len(recent_trends)
            older_success = sum(1 for t in older_trends if t['success']) / len(older_trends)
            
            if recent_success > older_success + 0.1:
                trend_analysis['success_rate'] = "improving"
            elif recent_success < older_success - 0.1:
                trend_analysis['success_rate'] = "declining"
            else:
                trend_analysis['success_rate'] = "stable"
        
        # Efficiency trend
        if older_trends:
            recent_efficiency = statistics.mean([t['efficiency_score'] for t in recent_trends])
            older_efficiency = statistics.mean([t['efficiency_score'] for t in older_trends])
            
            if recent_efficiency > older_efficiency + 0.05:
                trend_analysis['efficiency'] = "improving"
            elif recent_efficiency < older_efficiency - 0.05:
                trend_analysis['efficiency'] = "declining"
            else:
                trend_analysis['efficiency'] = "stable"
        
        return trend_analysis

    def _generate_recommendation(self, metrics: StrategyMetrics) -> str:
        """Generate a performance-based recommendation."""
        overall_score = metrics.calculate_overall_score()
        
        if overall_score >= 0.8:
            return "Excellent performance - continue using this strategy"
        elif overall_score >= 0.6:
            return "Good performance with room for optimization"
        elif overall_score >= 0.4:
            return "Moderate performance - consider improvements or alternatives"
        else:
            return "Poor performance - switch to alternative strategy recommended"

    def _calculate_confidence(self, metrics: StrategyMetrics) -> float:
        """Calculate confidence in the performance analysis."""
        # Base confidence on sample size
        sample_confidence = min(1.0, metrics.total_executions / 20)
        
        # Adjust for data recency
        recent_data_ratio = len(metrics.recent_performance_window) / metrics.total_executions
        recency_factor = min(1.0, recent_data_ratio * 2)
        
        # Adjust for consistency (lower variance = higher confidence)
        if len(metrics.completion_scores) > 1:
            score_variance = statistics.variance(metrics.completion_scores)
            consistency_factor = max(0.5, 1.0 - score_variance)
        else:
            consistency_factor = 0.5
        
        return (sample_confidence + recency_factor + consistency_factor) / 3

    def _generate_improvement_suggestions(self, metrics: StrategyMetrics) -> List[str]:
        """Generate specific improvement suggestions based on metrics."""
        suggestions = []
        
        if metrics.success_rate < self.thresholds.min_success_rate:
            suggestions.append("Improve error handling and recovery mechanisms")
        
        if metrics.average_iterations > self.thresholds.max_average_iterations:
            suggestions.append("Optimize iteration logic to reduce unnecessary steps")
        
        if metrics.average_execution_time_ms > self.thresholds.max_execution_time_ms:
            suggestions.append("Investigate performance bottlenecks and optimize execution speed")
        
        if metrics.error_rate > self.thresholds.max_error_rate:
            suggestions.append("Enhance input validation and error prevention")
        
        if metrics.average_efficiency_score < self.thresholds.min_efficiency_score:
            suggestions.append("Review resource usage and optimize strategy efficiency")
        
        if not suggestions:
            suggestions.append("Performance is within acceptable ranges")
        
        return suggestions

    def get_strategy_rankings(self) -> List[Tuple[str, float]]:
        """
        Get strategies ranked by overall performance.
        
        Returns:
            List of (strategy_name, overall_score) tuples sorted by performance
        """
        rankings = []
        
        for strategy_name, metrics in self.metrics_cache.items():
            if metrics.total_executions >= self.min_executions_for_analysis:
                overall_score = metrics.calculate_overall_score()
                rankings.append((strategy_name, overall_score))
        
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        # Update comparative rankings
        self.comparative_rankings.clear()
        for i, (strategy_name, _) in enumerate(rankings):
            self.comparative_rankings[strategy_name] = i + 1
        
        return rankings

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of overall performance monitoring status."""
        total_executions = sum(m.total_executions for m in self.metrics_cache.values())
        total_strategies = len(self.metrics_cache)
        
        if total_executions > 0:
            overall_success_rate = sum(
                m.successful_executions for m in self.metrics_cache.values()
            ) / total_executions
        else:
            overall_success_rate = 0.0
        
        return {
            'total_strategies_monitored': total_strategies,
            'total_executions_tracked': total_executions,
            'overall_success_rate': overall_success_rate,
            'active_executions': len(self.active_executions),
            'history_retention_days': self.history_retention_days,
            'strategy_rankings': self.get_strategy_rankings(),
            'monitoring_period_days': self.analysis_window_days
        }