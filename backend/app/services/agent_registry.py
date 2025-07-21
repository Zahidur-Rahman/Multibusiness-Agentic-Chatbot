from backend.app.services.general_agent import GeneralAgent
from backend.app.services.order_agent import OrderAgent
from backend.app.services.support_agent import SupportAgent

class AgentRegistry:
    def __init__(self):
        self._agents = {}

    def register_agent(self, name, agent_instance):
        self._agents[name] = agent_instance

    def get_agent(self, name):
        return self._agents.get(name)

# Instantiate and pre-register agents
agent_registry = AgentRegistry()
agent_registry.register_agent('general', GeneralAgent())
agent_registry.register_agent('order', OrderAgent())
agent_registry.register_agent('support', SupportAgent()) 