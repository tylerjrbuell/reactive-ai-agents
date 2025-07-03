import inspect
from functools import wraps
import docstring_parser
from typing import get_type_hints, get_origin, get_args


def _python_type_to_json_schema_type(python_type):
    """Convert Python type to JSON schema type string."""
    if python_type is None:
        return "string"  # Default fallback

    # Handle basic types
    if python_type in (str, type(None)):
        return "string"
    elif python_type in (int, float):
        return "number"
    elif python_type is bool:
        return "boolean"
    elif python_type in (list, tuple):
        return "array"
    elif python_type is dict:
        return "object"

    # Handle typing generics
    origin = get_origin(python_type)
    if origin is not None:
        if origin in (list, tuple):
            return "array"
        elif origin is dict:
            return "object"

    # Handle string representations
    if isinstance(python_type, str):
        python_type_lower = python_type.lower()
        if python_type_lower in ("str", "string"):
            return "string"
        elif python_type_lower in ("int", "integer", "float", "number"):
            return "number"
        elif python_type_lower in ("bool", "boolean"):
            return "boolean"
        elif python_type_lower in ("list", "array"):
            return "array"
        elif python_type_lower in ("dict", "object"):
            return "object"

    # Default fallback
    return "string"


def tool(description=None, parameters=None):
    """
    A decorator to convert a Python function into tool JSON metadata.

    :param description: Optional. Description of the function's purpose.
                        If not provided, extracted from the function's docstring.
    :param parameters: Optional. JSON schema for function parameters.
                        If not provided, inferred from the function's signature.
    :return: str: A decorator that adds the tool JSON metadata to the function.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # The function runs as usual
            return func(*args, **kwargs)

        # Parse docstring
        func_doc = docstring_parser.parse(func.__doc__) if func.__doc__ else None

        # Get function signature
        func_signature = inspect.signature(func)

        # Get type hints
        try:
            type_hints = get_type_hints(func)
        except (NameError, AttributeError):
            type_hints = {}

        # Build parameter properties
        properties = {}
        required = []

        for param_name, param in func_signature.parameters.items():
            # Skip *args and **kwargs
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue

            # Get type from type hints first
            param_type = type_hints.get(param_name)
            json_type = _python_type_to_json_schema_type(param_type)

            # Get description from docstring if available
            param_description = param_name
            if func_doc and func_doc.params:
                for doc_param in func_doc.params:
                    if doc_param.arg_name == param_name:
                        if doc_param.description:
                            param_description = doc_param.description
                        # Also try to get type from docstring if type hints failed
                        if json_type == "string" and doc_param.type_name:
                            json_type = _python_type_to_json_schema_type(
                                doc_param.type_name
                            )
                        break

            properties[param_name] = {
                "type": json_type,
                "description": param_description,
            }

            # Add to required if no default value
            if param.default is param.empty:
                required.append(param_name)

        # Build final parameters schema
        inferred_parameters = {
            "type": "object",
            "properties": properties,
            "required": required,
        }

        # Get description from docstring or decorator parameter
        final_description = description  # Use the decorator parameter first
        if not final_description and func_doc and func_doc.description:
            final_description = func_doc.description
        elif not final_description and func_doc and func_doc.short_description:
            final_description = func_doc.short_description

        # Attach the tool metadata as an attribute
        wrapper.__setattr__(
            "tool_definition",
            {
                "type": "function",
                "function": {
                    "name": func.__name__,
                    "description": final_description or f"Execute {func.__name__}",
                    "parameters": parameters or inferred_parameters,
                },
            },
        )
        return wrapper

    return decorator
