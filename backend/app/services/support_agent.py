from backend.app.services.base_agent import BaseAgent

class SupportAgent(BaseAgent):
    async def ainvoke(self, state):
        return {'response': 'SupportAgent: This is a stub.'} 