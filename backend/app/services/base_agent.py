from abc import ABC, abstractmethod

class BaseAgent(ABC):
    @abstractmethod
    async def ainvoke(self, state):
        """Run the agent's workflow asynchronously given the state."""
        pass 