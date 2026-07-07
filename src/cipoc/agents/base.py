from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from cipoc.llm import BaseAgentModel, LLMConfig, agent_model_for
from cipoc.utils import CipocConfig, load_config


class BaseAgent(ABC):
    _config: CipocConfig
    _llm_config: LLMConfig
    _tools: list[StructuredTool] | None
    agent: BaseAgentModel
    _graph: CompiledStateGraph
    _state: type[BaseModel]
    _input_schema: type[BaseModel]
    _output_schema: type[BaseModel]


    def __init__(
        self,
        agent_type: str | None = None,
        llm: BaseAgentModel | None = None,
        config: CipocConfig | None = None,
        **kwargs
    ) -> None:
        self._config = config or load_config()
        self._llm_config = self._config.llm_config(agent_type)
        self.agent = self._initialize_agent_model(llm, **kwargs)
        self._graph = self._build_graph()

    def _initialize_agent_model(self, llm: BaseAgentModel | None = None, **kwargs) -> BaseAgentModel:
        return llm or agent_model_for(self._llm_config.provider)(config=self._llm_config, **kwargs)
    
    def _build_graph(self) -> CompiledStateGraph:
        workflow = StateGraph(self._state, input_schema=self._input_schema, output_schema=self._output_schema)
        self._wire_graph(workflow)
        return workflow.compile()

    @abstractmethod
    def _wire_graph(self, workflow: StateGraph) -> None:
        ...

    @abstractmethod
    def run(self) -> Any:
        ... 

    def draw(self, path: str) -> None:
        """Render the compiled graph. Writes a PNG where possible (needs network),
        otherwise prints the ASCII diagram so the CLI stays self-contained."""
        graph = self._graph.get_graph()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        try:
            graph.draw_mermaid_png(output_file_path=path)
        except Exception:
            print(graph.draw_ascii())
