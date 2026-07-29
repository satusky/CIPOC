from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import StructuredTool
from langchain_core.runnables.graph import CurveStyle, NodeStyles
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from cipoc.llm import BaseAgentModel, LLMConfig, RetryPolicy, agent_model_for
from cipoc.utils import CipocConfig, arun_with_progress, load_config


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
    _retry_policy: RetryPolicy
    _tools: list[StructuredTool] | None
    agent: BaseAgentModel
    _graph: CompiledStateGraph
    _state: type[BaseModel]
    _input_schema: type[BaseModel]
    _output_schema: type[BaseModel]
    #: Whether this instance wired the async node twins. Carries a real default,
    #: not just an annotation: ``tests/fake_orchestrator.py`` builds an agent with
    #: ``object.__new__`` and never runs ``__init__``, and ``_wire_graph`` reads it.
    _async: bool = False


    def __init__(
        self,
        agent_type: str | None = None,
        llm: BaseAgentModel | None = None,
        config: CipocConfig | None = None,
        *,
        use_async: bool | None = None,
        **kwargs
    ) -> None:
        self._config = config or load_config()
        # Read through agent_settings, never llm_config: OpenAIConfig is
        # extra="allow", so an `async_mode:` key routed the other way would land
        # in ChatOpenAI(async_mode=...). An explicit kwarg wins over config so a
        # caller — or the orchestrator propagating its mode to subagents — is
        # never silently overridden by a per-agent block.
        settings = self._config.agent_settings(agent_type)
        self._async = bool(settings.get("async_mode", False)) if use_async is None else bool(use_async)
        self._llm_config = self._config.llm_config(agent_type)
        self._retry_policy = self._config.retry_policy(agent_type)
        self.agent = self._initialize_agent_model(llm, **kwargs)
        self._graph = self._build_graph()

    @property
    def retry_policy(self) -> RetryPolicy:
        """Retry policy for this agent's LLM-backed nodes.

        Pass to ``add_node(..., retry_policy=self.retry_policy)`` on every node
        that calls the model, and only those. Retrying at the node that issued
        the request replays one LLM call rather than a whole branch, and keeps
        the retry off deterministic nodes where a failure is a real bug.

        Subgraph nodes carry their own policy, so a node whose body invokes a
        subagent graph must not also be wrapped — that would multiply attempts.
        """
        return self._retry_policy

    def _initialize_agent_model(self, llm: BaseAgentModel | None = None, **kwargs) -> BaseAgentModel:
        return llm or agent_model_for(self._llm_config.provider)(config=self._llm_config, **kwargs)

    def _node(self, name: str) -> Callable:
        """The callable to register under node ``name``: the ``a``-prefixed async
        twin in async mode, the sync method otherwise.

        Only the LLM-calling nodes have twins; everything else is deterministic
        and registers unchanged in both graphs, which is why the lookup falls back
        to the sync method rather than requiring one. LangGraph runs a sync node
        under ``ainvoke`` in an executor, fine for CPU-light bookkeeping.

        The ``retry_policy=`` kwarg is orthogonal — this resolves *which* callable
        a node runs, not how it retries — so the retry wiring rules in CLAUDE.md
        hold identically in both graphs.
        """
        if self._async:
            return getattr(self, f"a{name}", getattr(self, name))
        return getattr(self, name)

    def _require_mode(self, want_async: bool) -> None:
        """Fail fast when ``run()`` is called on an async-wired agent, or vice versa.

        The graph is wired once at construction, and neither mismatch reports
        itself well. ``invoke`` over coroutine nodes raises a LangGraph
        ``TypeError`` naming one node rather than the agent; ``ainvoke`` over sync
        nodes does not fail at all — it runs every node in an executor, quietly
        delivering none of the concurrency the async mode was chosen for.
        """
        if self._async == want_async:
            return
        called, built, other = (
            ("arun()", "sync", "run()") if want_async else ("run()", "async", "arun()")
        )
        raise RuntimeError(
            f"{type(self).__name__}.{called} was called on a {built}-mode agent. "
            f"Call {other} instead, or rebuild the agent with use_async={want_async}."
        )

    async def _arun_graph(
        self, graph_input: Any, *, progress: bool = False, **progress_kwargs
    ) -> Any:
        """Drive the compiled graph on the caller's loop. Every ``arun`` goes
        through here so the mode guard is applied in exactly one place.

        ``progress`` paints the same dashboard the sync path does, driven by
        ``graph.astream`` instead of ``graph.stream``; ``progress_kwargs`` are the
        display settings (description, subgraphs, target groups) each agent's
        ``run`` already passes, and are ignored when progress is off.
        """
        self._require_mode(True)
        if not progress:
            return await self._graph.ainvoke(graph_input)
        return await arun_with_progress(self._graph, graph_input, **progress_kwargs)

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

    @abstractmethod
    async def arun(self) -> Any:
        """Async twin of :meth:`run`. Deliberately a coroutine with no internal
        ``asyncio.run``: the caller owns the loop, so this stays usable from a
        Databricks/Jupyter cell that already has one running. ``run`` remains the
        notebook-safe default.
        """
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
