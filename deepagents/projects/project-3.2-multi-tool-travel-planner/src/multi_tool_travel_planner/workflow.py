from __future__ import annotations

from typing import TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .config import Settings


def search_flights(origin: str, destination: str) -> list[str]:
    return [
        f"Flight A: {origin} -> {destination}, 320 USD",
        f"Flight B: {origin} -> {destination}, 410 USD",
    ]


def search_hotels(destination: str, nights: int) -> list[str]:
    return [
        f"Hotel Central in {destination}: 110 USD/night for {nights} nights",
        f"Hotel Riverside in {destination}: 150 USD/night for {nights} nights",
    ]


def estimate_budget(flights: list[str], hotels: list[str], budget_cap: int, nights: int) -> str:
    cheapest_flight = 320
    cheapest_hotel = 110 * nights
    local_transport = 25 * nights
    total = cheapest_flight + cheapest_hotel + local_transport
    status = "within" if total <= budget_cap else "over"
    return f"Estimated total is {total} USD, which is {status} the budget cap of {budget_cap} USD."


def calculate_total_estimated_cost(nights: int) -> float:
    return float(320 + (110 * nights) + (25 * nights))


class TravelState(TypedDict, total=False):
    origin: str
    destination: str
    nights: int
    budget: float
    flights: tuple[str, ...]
    hotels: tuple[str, ...]
    budget_note: str
    total_estimated_cost: float
    budget_retry_count: int
    itinerary: str


def fetch_options(state: TravelState) -> TravelState:
    parallel_fetcher = RunnableParallel(
        flights=RunnableLambda(lambda payload: search_flights(payload["origin"], payload["destination"])),
        hotels=RunnableLambda(lambda payload: search_hotels(payload["destination"], payload["nights"])),
    )
    results = parallel_fetcher.invoke(state)
    if state.get("budget_retry_count", 0) > 0:
        return {
            "flights": tuple(results["flights"][:1]),
            "hotels": tuple(results["hotels"][:1]),
        }
    return {
        "flights": tuple(results["flights"]),
        "hotels": tuple(results["hotels"]),
    }


def route_after_budget(state: TravelState) -> str:
    if state["total_estimated_cost"] > state["budget"] and state.get("budget_retry_count", 0) < 1:
        return "replan"
    return "build_itinerary"


def build_app(settings: Settings):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You build concise travel itineraries using the provided options and budget note."),
            (
                "human",
                "Origin: {origin}\nDestination: {destination}\nNights: {nights}\n\nFlights:\n{flights}\n\nHotels:\n{hotels}\n\nBudget:\n{budget_note}",
            ),
        ]
    )
    model = ChatOpenAI(model=settings.model, temperature=settings.temperature)
    planner = prompt | model | StrOutputParser()

    def fetch_options_node(state: TravelState) -> TravelState:
        return fetch_options(state)

    def budget_node(state: TravelState) -> TravelState:
        total_cost = calculate_total_estimated_cost(state["nights"])
        return {
            "total_estimated_cost": total_cost,
            "budget_note": estimate_budget(
                list(state.get("flights", ())),
                list(state.get("hotels", ())),
                int(state["budget"]),
                state["nights"],
            ),
            "budget_retry_count": state.get("budget_retry_count", 0) + (1 if total_cost > state["budget"] else 0),
        }

    def itinerary_node(state: TravelState) -> TravelState:
        itinerary = planner.invoke(
            {
                "origin": state["origin"],
                "destination": state["destination"],
                "nights": state["nights"],
                "flights": "\n".join(state.get("flights", ())),
                "hotels": "\n".join(state.get("hotels", ())),
                "budget_note": state.get("budget_note", "No budget note available."),
            }
        )
        return {"itinerary": itinerary}

    graph = StateGraph(TravelState)
    graph.add_node("fetch_options", fetch_options_node)
    graph.add_node("budget", budget_node)
    graph.add_node("itinerary", itinerary_node)
    graph.add_edge(START, "fetch_options")
    graph.add_edge("fetch_options", "budget")
    graph.add_conditional_edges("budget", route_after_budget, {"replan": "fetch_options", "build_itinerary": "itinerary"})
    graph.add_edge("itinerary", END)
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
