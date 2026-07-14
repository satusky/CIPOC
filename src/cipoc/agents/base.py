from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import StructuredTool
from langchain_core.runnables.graph import CurveStyle, NodeStyles
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from cipoc.llm import BaseAgentModel, LLMConfig, agent_model_for
from cipoc.utils import CipocConfig, load_config


# House style for graph PNGs. font-family must ride in the NodeStyles strings:
# langgraph wraps node labels in HTML <p> tags, which ignore themeVariables.fontFamily.
# The end-node fill is blue-550 (#1c5cab), not #2a78d6 — white text on the brighter
# blue fails WCAG AA contrast (4.42:1).
MERMAID_STYLE: dict[str, Any] = dict(
    curve_style=CurveStyle.BASIS,
    node_colors=NodeStyles(
        default="fill:#cde2fb,stroke:#2a78d6,stroke-width:1.5px,color:#0b0b0b,font-family:sans-serif,line-height:1.2",
        first="fill:#fcfcfb,stroke:#898781,stroke-width:1.5px,color:#52514e,font-family:sans-serif",
        last="fill:#1c5cab,stroke:#104281,stroke-width:1.5px,color:#ffffff,font-family:sans-serif",
    ),
    background_color="#fcfcfb",
    padding=20,
    frontmatter_config={
        "config": {
            "theme": "base",
            "themeVariables": {
                "lineColor": "#898781",
                "clusterBkg": "#f0efec",
                "clusterBorder": "#c3c2b7",
                "titleColor": "#52514e",
                "edgeLabelBackground": "#fcfcfb",
                "fontSize": "15px",
            },
            "flowchart": {"curve": "basis", "nodeSpacing": 40, "rankSpacing": 45},
        }
    },
)


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

    def draw(self, path: str, **mermaid_kwargs) -> None:
        """Render the compiled graph. Writes a PNG where possible (needs network),
        otherwise prints the ASCII diagram so the CLI stays self-contained.

        Styled with ``MERMAID_STYLE``; pass any ``draw_mermaid_png`` keyword to
        override individual settings.
        """
        graph = self._graph.get_graph(xray=True)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        try:
            graph.draw_mermaid_png(output_file_path=path, **{**MERMAID_STYLE, **mermaid_kwargs})
        except Exception:
            print(graph.draw_ascii())
