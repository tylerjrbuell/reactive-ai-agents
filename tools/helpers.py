import inspect
from functools import wraps
import docstring_parser
from typing import get_type_hints


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

        # Infer function metadata
        func_doc = docstring_parser.parse(func.__doc__)
        # Infer parameters from docstring
        doc_inferred_properties = {
            p.arg_name: {
                "type": p.type_name,
                "description": p.description or p.arg_name,
            }
            for p in func_doc.params
        }
        # Infer parameters from function signature if docstring not provided
        func_signature = inspect.signature(func)
        sig_inferred_properties = {
            p.name: {
                "type": get_type_hints(func).get(p.name, "any").__name__,
                "description": p.name or p,
            }
            for p in func_signature.parameters.values()
        }
        inferred_parameters = {
            "type": "object",
            "properties": doc_inferred_properties or sig_inferred_properties,
            "required": list(func_signature.parameters.keys()),
        }

        # Attach the Ollama tool metadata as an attribute
        wrapper.__setattr__(
            "tool_definition",
            {
                "type": "function",
                "function": {
                    "name": func.__name__,
                    "description": description or func_doc.description,
                    "parameters": parameters or inferred_parameters,
                },
            },
        )
        return wrapper

    return decorator
