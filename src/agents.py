"""Specialized agents and orchestration for the document assistant."""

from dataclasses import dataclass
from typing import Protocol, Sequence

from src.guardrails import SafetyCheck, validate_final_answer, validate_question
from src.rag_pipeline import GroundedAnswer, OpenAIAnswerGenerator
from src.retriever import RetrievedPassage, SearchableVectorStore, retrieve_passages


@dataclass(frozen=True)
class QuestionPlan:
    """A small, inspectable plan produced before retrieval begins."""

    original_question: str
    search_query: str
    objective: str
    top_k: int


@dataclass(frozen=True)
class AgentStep:
    """One completed stage in an orchestrated run."""

    agent: str
    status: str
    detail: str


@dataclass(frozen=True)
class AgenticAnswer:
    """Final result, evidence, and an auditable agent trace."""

    answer: GroundedAnswer
    passages: tuple[RetrievedPassage, ...]
    plan: QuestionPlan
    steps: tuple[AgentStep, ...]
    safety_checks: tuple[SafetyCheck, ...]


class AnswerGenerator(Protocol):
    """Minimal answer-generator interface used by the reasoning agent."""

    def generate(
        self, question: str, passages: Sequence[RetrievedPassage]
    ) -> GroundedAnswer: ...


class PlanningAgent:
    """Normalize a request and define a bounded retrieval objective."""

    name = "Planning agent"

    def plan(self, question: str, top_k: int) -> QuestionPlan:
        normalized, _ = validate_question(question)
        if not 1 <= top_k <= 8:
            raise ValueError("top_k must be between 1 and 8.")
        return QuestionPlan(
            original_question=normalized,
            search_query=normalized,
            objective="Find document passages that directly support the answer.",
            top_k=top_k,
        )


class RetrievalAgent:
    """Execute semantic retrieval from the indexed document collection."""

    name = "Retrieval agent"

    def retrieve(
        self, vector_store: SearchableVectorStore, plan: QuestionPlan
    ) -> tuple[RetrievedPassage, ...]:
        passages = retrieve_passages(
            vector_store, plan.search_query, top_k=plan.top_k
        )
        if not passages:
            raise ValueError("No relevant passages were found in the document index.")
        return passages


class ReasoningAgent:
    """Create an evidence-only response from retrieved passages."""

    name = "Reasoning agent"

    def __init__(self, generator: AnswerGenerator) -> None:
        self._generator = generator

    def reason(
        self, question: str, passages: Sequence[RetrievedPassage]
    ) -> GroundedAnswer:
        return self._generator.generate(question, passages)


class ValidationAgent:
    """Perform a final deterministic check before an answer is displayed."""

    name = "Validation agent"

    def validate(
        self, answer: GroundedAnswer, passages: Sequence[RetrievedPassage]
    ) -> tuple[SafetyCheck, ...]:
        return validate_final_answer(answer, passages)


class AgentOrchestrator:
    """Coordinate planning, retrieval, reasoning, and validation in order."""

    def __init__(
        self,
        generator: AnswerGenerator,
        planning_agent: PlanningAgent | None = None,
        retrieval_agent: RetrievalAgent | None = None,
        validation_agent: ValidationAgent | None = None,
    ) -> None:
        self.planning_agent = planning_agent or PlanningAgent()
        self.retrieval_agent = retrieval_agent or RetrievalAgent()
        self.reasoning_agent = ReasoningAgent(generator)
        self.validation_agent = validation_agent or ValidationAgent()

    def run(
        self,
        question: str,
        vector_store: SearchableVectorStore,
        top_k: int = 4,
    ) -> AgenticAnswer:
        steps: list[AgentStep] = []

        normalized_question, input_checks = validate_question(question)
        plan = self.planning_agent.plan(normalized_question, top_k)
        steps.append(
            AgentStep(
                self.planning_agent.name,
                "complete",
                f"Prepared a focused search for up to {plan.top_k} passages.",
            )
        )

        passages = self.retrieval_agent.retrieve(vector_store, plan)
        steps.append(
            AgentStep(
                self.retrieval_agent.name,
                "complete",
                f"Retrieved {len(passages)} ranked evidence passage(s).",
            )
        )

        answer = self.reasoning_agent.reason(plan.original_question, passages)
        steps.append(
            AgentStep(
                self.reasoning_agent.name,
                "complete",
                "Generated an answer restricted to retrieved evidence.",
            )
        )

        output_checks = self.validation_agent.validate(answer, passages)
        steps.append(
            AgentStep(
                self.validation_agent.name,
                "complete",
                "Verified the final response and its source labels.",
            )
        )
        return AgenticAnswer(
            answer,
            passages,
            plan,
            tuple(steps),
            input_checks + output_checks,
        )


def run_agent_workflow(
    question: str,
    vector_store: SearchableVectorStore,
    api_key: str,
    model: str = "gpt-5-mini",
    top_k: int = 4,
) -> AgenticAnswer:
    """Create the production generator and run all specialized agents."""
    generator = OpenAIAnswerGenerator(api_key=api_key, model=model)
    return AgentOrchestrator(generator).run(question, vector_store, top_k=top_k)
