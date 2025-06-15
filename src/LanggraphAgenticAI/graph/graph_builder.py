from langgraph.graph import StateGraph, START,END, MessagesState
from langgraph.prebuilt import tools_condition,ToolNode
from langchain_core.prompts import ChatPromptTemplate
from src.LanggraphAgenticAI.state.state import State
from src.LanggraphAgenticAI.nodes.basic_chatbot_node import BasicChatbotNode
from src.LanggraphAgenticAI.nodes.chatbot_with_Tool_node import ChatbotWithToolNode
from src.LanggraphAgenticAI.tools.search_tool import get_tools,create_tool_node
from src.LanggraphAgenticAI.nodes.ai_news_node import AINewsNode




class GraphBuilder:

    def __init__(self, model):
        self.llm = model

    def basic_chatbot_build_graph(self, graph_builder):
        chatbot_node = BasicChatbotNode(self.llm).process
        graph_builder.add_node("chatbot", chatbot_node)
        graph_builder.set_entry_point("chatbot")
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("chatbot", END)

    def chatbot_with_tools_build_graph(self, graph_builder):
        tools = get_tools()
        tool_node = create_tool_node(tools)
        chatbot_node = ChatbotWithToolNode(self.llm).create_chatbot(tools)

        graph_builder.add_node("chatbot", chatbot_node)
        graph_builder.add_node("tools", tool_node)
        graph_builder.set_entry_point("chatbot")
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_conditional_edges("chatbot", tools_condition)
        graph_builder.add_edge("tools", "chatbot")

    def ai_news_build_graph(self, graph_builder):
        ai_news_node = AINewsNode(self.llm)
        graph_builder.add_node("fetch_news", ai_news_node.fetch_news)
        graph_builder.add_node("summarize_news", ai_news_node.summarize_news)
        graph_builder.add_node("save_result", ai_news_node.save_result)
        graph_builder.set_entry_point("fetch_news")
        graph_builder.add_edge(START, "fetch_news")
        graph_builder.add_edge("fetch_news", "summarize_news")
        graph_builder.add_edge("summarize_news", "save_result")
        graph_builder.add_edge("save_result", END)

    def setup_graph(self, usecase: str):
        graph_builder = StateGraph(State)  # ✅ NEW INSTANCE every time

        if usecase == "Basic Chatbot":
            self.basic_chatbot_build_graph(graph_builder)
        elif usecase == "Chatbot with Tool":
            self.chatbot_with_tools_build_graph(graph_builder)
        elif usecase == "AI News":
            self.ai_news_build_graph(graph_builder)
        else:
            raise ValueError(f"Unknown usecase: {usecase}")

        return graph_builder.compile()
