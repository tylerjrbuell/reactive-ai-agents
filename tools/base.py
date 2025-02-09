from typing import Any, Callable


class Tool:
    def __init__(self, function: Callable):
        """
        Initialize a Tool instance.

        :param function: The function that the tool will execute.
        """
        self.function = function
        self.name = function.__name__
        self.tool_definition = function.tool_definition

    async def use(self, params: dict) -> Any:
        """
        Use the tool by executing the tool function with the given parameters.

        :param params: The parameters to pass to the tool's function.
        :return: The result of the tool's execution.
        """
        try:
            result = await self.function(**params)
            return result
        except Exception as e:
            return f"Tool Execution Error: {e}"
