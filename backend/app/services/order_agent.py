from backend.app.services.base_agent import BaseAgent

class OrderAgent(BaseAgent):
    async def ainvoke(self, state):
        return {'response': 'OrderAgent: This is a stub.'} 