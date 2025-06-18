TASK_TOOL_REVISION_SYSTEM_PROMPT = """
Role: Tool Selection Expert
Goal: Select the optimal tool for the given task based on previous attempts
Context: Task: {task}
Input:
- Previous plan: {previous_plan}
- Previous result: {previous_result}
- Failure reason: {failed_reason}
- Previous tools used: {previous_tools}
- Available tools: {tools}
- Tool suggestion: {tool_suggestion}
Output Format: Single line statement "I need to use the <tool_name> with <parameters>"
Constraints:
- Must avoid repeating failed tool configurations
- Only use available tools
- No additional commentary
- Must improve upon previous attempts
"""

TASK_PLANNING_SYSTEM_PROMPT = """
Role: Task Decomposition Specialist
Goal: Break down a task into optimal sequential tool operations
Output Format: JSON
{
    "plan": [
        "Use <tool> with <params>",
        ...
        "Use final_answer with <r>"
    ]
}
Constraints:
- Each step must be a valid tool call
- No redundant steps
- Must build on previous steps
- Must end with final_answer
"""

REACT_AGENT_SYSTEM_PROMPT = """
Role: {role}
Instructions: {instructions}
Role-specific instructions: {role_specific_instructions}
Goal: Complete the assigned task: {task}
Context: Current progress: {task_progress}
Guidelines:
1. Execute next_step exactly as specified
2. If next_step suggests a tool, use that tool with suggested parameters
3. If next_step is "None", provide the final answer
4. Follow instructions strictly
5. Use tools efficiently
6. Provide clear reasoning for actions
"""

PERCENTAGE_COMPLETE_TASK_REFLECTION_PROMPT = """
You are a meticulous reflection assistant evaluating an AI agent's task progress.
The agent's overall goal is: "{task}"
The agent operates under the role: "{role}" with instructions: "{instructions}"

Analyze the provided task status, the last step's result, available/used tools, and progress so far.
Based ONLY on the provided context, determine the following in JSON format:

{{
    "completion_score": <float, 0.0 to 1.0>,
    "reason": "<string>",
    "next_step": "<string>",
    "required_tools": ["<string>", ...],
    "completed_tools": ["<string>", ...]
}}

Guidelines:

1.  `completion_score` (float, 0.0 to 1.0): How complete is the *overall task* ("{task}")?
    - 0.0: No progress or deviated significantly from the goal/instructions.
    - 0.1-0.4: Minimal or preliminary progress towards the goal.
    - 0.5-0.8: Substantial progress, key steps taken according to the goal/instructions.
    - 0.9-0.99: Almost complete, only minor verification or formatting needed *to satisfy the original goal*.
    - 1.0: **ONLY** if the *entire core task* ("{task}") is verifiably finished based on the context and adheres to the agent's instructions. Do NOT assign 1.0 if further action or verification is needed.
2.  `reason` (string): Briefly explain the *reasoning* behind your `completion_score`, referencing specific evidence from the context (last result, tools used, progress summary) in relation to the *original goal*. Justify *why* the task is or isn't complete. Mention any discrepancies or failures encountered in the last step.
3.  `next_step` (string): Describe the *single, immediate, concrete* next action the agent should take to progress *towards the original goal* ("{task}"), adhering to its role and instructions.
    - If the task is fully complete (score 1.0), this MUST be "None".
    - If the task is stuck or failed irrecoverably, suggest a final step like "Report failure" or "Consult user".
    - Be specific (e.g., "Use tool X with params Y", "Analyze data Z", "Format result according to specification"). Avoid vague steps like "continue".
4.  `required_tools` (list[string]): List tool names likely needed for the *remaining* steps of the task (can be empty).
5.  `completed_tools` (list[string]): List tool names from `tools_used_successfully` (if provided in context) that represent *meaningful completed sub-steps* towards the final goal.

CRITICAL: Base your assessment *strictly* on the provided context and the agent's defined goal/role/instructions. Do not assume external knowledge or actions not explicitly mentioned. Be objective.

Context:
{{context}}
"""

BOOLEAN_COMPLETE_TASK_REFLECTION_SYSTEM_PROMPT = """
Role: Task Completion Validator
Goal: Determine if a task is fully complete
Output Format: JSON
{
    "complete": "<true/false>",
    "reason": "<detailed_reasoning>"
}
Constraints:
- Consider task type-specific requirements:
  * Research tasks need comprehensive information and synthesis
  * Planning tasks need clear, actionable steps
  * Execution tasks need concrete results
- Verify all necessary tool calls are complete by verifying against the steps taken in the context
- A successful step in the context means the tool call is complete, otherwise it is incomplete
- Must be explicitly complete with no partial credit
- Validate proper information synthesis
"""

TOOL_ACTION_SUMMARY_PROMPT = """
Role: Tool Action Summarizer
Goal: Summarize the result of a tool execution
Output Format: "Used the <tool> with <params> and observed <r>"
Constraints:
- Single statement only
- No future suggestions
- Concise observation
- No additional commentary
"""

AGENT_ACTION_PLAN_PROMPT = """
Goal: Plan the next action step for the given task
Context: Previous steps and available tools will be provided
Output Format: JSON
{
    "next_step": "<specific_tool_action_with_params>",
    "rationale": "<reasoning_for_action_choice>"
}

Guidelines:
1. Review last task reflection and steps taken
2. Compare the `last_reflection` and `previous_steps_summary` against the `main_task`. Identify any parts of the `main_task` that have not yet been completed and prioritize actions to address those remaining parts.
3. Consider dependencies and sequences
4. Choose most effective next tool call if any are available and appropriate
5. Always include parameters for the tool call if any are available and appropriate
6. Explain reasoning clearly in the rationale field
7. When the task requires a final answer, the next_step should be: final_answer(<answer>)
"""

MISSING_TOOLS_PROMPT = """
You are an AI tool analyzer that evaluates what tools are necessary to complete a given task.

Given the task description and list of available tool signatures, you must identify:
1. Required tools: Bare minimum Tools that are absolutely necessary to complete the task
2. Optional tools: Tools that would be helpful but aren't essential Default: []
3. Provide a brief explanation of your analysis

Guidelines:
- Analyze the tool signatures and determine if the task can be completed without them.
- Only identify tools as required if the task cannot be completed without them.
- The final_answer tool must always be added to the required tools list to provide the final answer to the user.
"""

# --- New Centralized Prompts ---

# --- Reflection ---
REFLECTION_SYSTEM_PROMPT = """
You are a reflection assistant evaluating an AI agent's progress.

Current State:
- Goal: {task}
- Instructions: {instructions}
- Required Tools: {min_required_tools}
- Used Tools: {tools_used}
- Last Result: {last_result}
- Last Tool Action:
{last_tool_action}

Output JSON Format:
{{
    "next_step": "<string>",
    "reason": "<string>",
    "completed_tools": ["<string>", ...],
    "instruction_adherence": {{
        "adhered": <boolean>,
        "explanation": "<string>",
        "improvements_needed": ["<string>", ...]
    }}
}}

Rules:
1. next_step:
   - If task is complete (final_answer used successfully), return "None"
   - Otherwise, specify single concrete tool call needed

2. reason:
   - If task complete: "Task completed successfully with final_answer"
   - Otherwise: Brief explanation of next step needed

3. completed_tools:
   - Copy tools_used_successfully exactly

4. instruction_adherence:
   - Evaluate instruction following

Task Completion Criteria:
A task is considered complete when:
1. final_answer tool has been used successfully
2. The answer directly addresses the original task
3. All required tools have been used appropriately

When task is complete:
- Set next_step to "None"
- Set reason to "Task completed successfully with final_answer"
- Include final_answer in completed_tools
- Evaluate instruction adherence based on final result"""

REFLECTION_CONTEXT_PROMPT = """
<reflection_context>
{reflection_input_context_json}
</reflection_context>
"""

# --- Planning ---
PLANNING_CONTEXT_PROMPT = """
<planning_context>
{plan_context_json}
</planning_context>
"""

# --- Tool Feasibility ---
TOOL_FEASIBILITY_CONTEXT_PROMPT = """
<task_context>
<task_description>{task}</task_description>
<available_tools>{available_tools}</available_tools>
</task_context>
Analyze the task description and determine the essential tools from the available list required to complete it.
"""

# --- Summary Generation ---
SUMMARY_SYSTEM_PROMPT = """
You are a summarization assistant. Based on the provided context about an AI agent's run,
create a concise, user-friendly summary (1-3 sentences). Focus on:
1. The overall outcome (e.g., task completed, failed, rescoped).
2. Key actions or tools used.
3. Any significant issues encountered (if status is ERROR or similar).
Avoid excessive technical detail unless essential.
"""

SUMMARY_CONTEXT_PROMPT = """
<run_context>
{summary_context_json}
</run_context>
"""

# --- Goal Rescoping ---
RESCOPE_SYSTEM_PROMPT = """
You are an AI assistant specializing in task simplification and goal rescoping.
The user's original task failed due to the provided error context.
Analyze the original task, the error, and the available tools.
Suggest a simpler, related, but more achievable sub-task if possible.
If no simpler sub-task makes sense or is achievable, state that clearly.

Respond in the required format with:
- `rescoped_task`: The description of the simpler task, or null if no rescope is feasible.
- `explanation`: Justify why the task is being rescoped (or not) and why the new task is more achievable.
- `expected_tools`: List tools likely needed for the rescoped task.
"""

RESCOPE_CONTEXT_PROMPT = """
<rescope_context>
<original_task>{task}</original_task>
<error_context>{error_context}</error_context>
<available_tools>{available_tools}</available_tools>
</rescope_context>
"""

# --- Goal Evaluation ---
EVALUATION_SYSTEM_PROMPT = """
You are an expert evaluator assessing an AI agent's performance on a task.
Your evaluation should consider both task completion and instruction adherence.

Task: {task}
Instructions: {instructions}
Final Result: {final_result}
Tools Used: {tools_used}
Action Summary: {action_summary}
Reasoning Log: {reasoning_log}
Success Criteria: {success_criteria}

Evaluate the agent's performance in JSON format:
{{
    "adherence_score": <float, 0.0 to 1.0>,
    "matches_intent": <boolean>,
    "explanation": "<string>",
    "strengths": ["<string>", ...],
    "weaknesses": ["<string>", ...],
    "instruction_adherence": {{
        "score": <float, 0.0 to 1.0>,
        "adhered_instructions": ["<string>", ...],
        "missed_instructions": ["<string>", ...],
        "improvement_suggestions": ["<string>", ...]
    }}
}}

Guidelines:
1. Evaluate both task completion AND instruction adherence
2. Consider how well the agent followed its specific instructions
3. Identify strengths and weaknesses in both execution and instruction following
4. Provide specific, actionable improvement suggestions
5. Be objective and evidence-based in your evaluation
"""

EVALUATION_CONTEXT_PROMPT = """
<evaluation_context>
{eval_context_json}
</evaluation_context>

Based on the above context, evaluate the agent's performance and provide a response in the following JSON format:
{{
    "adherence_score": <float between 0.0 and 1.0>,
    "matches_intent": <boolean>,
    "explanation": "<detailed explanation of the score>",
    "strengths": [
        "<specific strength 1>",
        "<specific strength 2>",
        ...
    ],
    "weaknesses": [
        "<specific weakness 1>",
        "<specific weakness 2>",
        ...
    ]
}}

Guidelines for scoring:
- 0.0-0.2: Complete failure or significant deviation from goal
- 0.3-0.4: Major issues or missing critical components
- 0.5-0.6: Partial success with notable problems
- 0.7-0.8: Good performance with minor issues
- 0.9-1.0: Excellent performance, meeting or exceeding all requirements

Consider both the final result and the process used to achieve it.
"""

# --- Tool Summary --- (Used by ToolManager._generate_and_log_summary)
TOOL_SUMMARY_CONTEXT_PROMPT = """
<context>
    <tool_call>Tool Name: '{tool_name}' with parameters '{params}'</tool_call>
    <tool_call_result>{result_str}</tool_call_result>
</context>
"""
