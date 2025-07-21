from backend.app.services.base_agent import BaseAgent
from backend.app.services.chat_graph import chat_graph

class GeneralAgent(BaseAgent):
    async def ainvoke(self, state):
        return await chat_graph.ainvoke(state) 