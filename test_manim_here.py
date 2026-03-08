"""
Traditional RAG vs Agentic RAG – Manim 0.19
Run:
    manim -qh rag_comparison.py RAGComparison
"""

import os
from dotenv import load_dotenv
from manim import *
from elevenlabs.client import ElevenLabs
from elevenlabs import save

load_dotenv()

VOICE_ID = "pNInz6obpgDQGcFmaJgB"

NARRATIONS = {
    "intro": "Let's compare Traditional RAG and Agentic RAG.",
    "trad_query": "Traditional RAG starts with a user query.",
    "trad_embed": "The query becomes an embedding.",
    "trad_search": "Vector search retrieves documents.",
    "trad_llm": "Documents are passed to the language model.",
    "trad_answer": "The model produces the final answer.",
    "agent_query": "Agentic RAG also starts with a query.",
    "agent_reason": "An agent reasons about the problem.",
    "agent_tools": "The agent chooses tools.",
    "agent_context": "Multiple retrievals build context.",
    "agent_llm": "Context is given to the language model.",
    "agent_answer": "The system produces the final answer.",
}

os.makedirs("voices_cmp", exist_ok=True)


def tts(key):
    api_key = os.getenv("ELEVEN_API_KEY")
    if not api_key:
        return None

    path = f"voices_cmp/{key}.mp3"
    if os.path.exists(path):
        return path

    client = ElevenLabs(api_key=api_key)

    audio = client.text_to_speech.convert(
        voice_id=VOICE_ID,
        model_id="eleven_multilingual_v2",
        text=NARRATIONS[key],
    )

    save(audio, path)
    return path


for k in NARRATIONS:
    tts(k)


# ─────────────────────────────────────────────
# BLACK & WHITE THEME
# ─────────────────────────────────────────────

BG = "#000000"
WHITE = "#FFFFFF"
GREY = "#777777"
PANEL_BG = "#111111"


# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────

def make_box(label, w=2.4, h=0.6):
    rect = RoundedRectangle(
        corner_radius=0.15,
        width=w,
        height=h,
        fill_color=PANEL_BG,
        fill_opacity=1,
        stroke_color=WHITE,
        stroke_width=2
    )

    text = Text(label, font_size=20, color=WHITE).move_to(rect)

    return VGroup(rect, text)


def arrow(a, b):
    return Arrow(
        start=a.get_right(),
        end=b.get_left(),
        buff=0.25,
        color=WHITE,
        stroke_width=2
    )


def v_arrow(a, b):
    return Arrow(
        start=a.get_bottom(),
        end=b.get_top(),
        buff=0.25,
        color=WHITE,
        stroke_width=2
    )


def subtitle(text):
    return Text(
        text,
        font_size=22,
        color=WHITE,
        slant=ITALIC
    )


# ─────────────────────────────────────────────
# Scene
# ─────────────────────────────────────────────

class RAGComparison(Scene):

    def setup(self):
        self.camera.background_color = BG

    def play_voice(self, key):
        path = f"voices_cmp/{key}.mp3"
        if os.path.exists(path):
            self.add_sound(path)

    def show_sub(self, text, duration=2):
        s = subtitle(text).to_edge(DOWN)

        self.play(FadeIn(s, shift=UP * 0.2))
        self.wait(duration)
        self.play(FadeOut(s))

    # ─────────────────────────────────────────
    # Intro
    # ─────────────────────────────────────────

    def intro(self):

        title = Text(
            "Traditional RAG vs Agentic RAG",
            font_size=52,
            color=WHITE
        )

        self.play(Write(title))
        self.wait(2)

        self.play(FadeOut(title))

    # ─────────────────────────────────────────
    # Traditional RAG
    # ─────────────────────────────────────────

    def traditional(self):

        boxes = [
            make_box("User Query"),
            make_box("Embedding"),
            make_box("Vector Search"),
            make_box("Docs"),
            make_box("LLM"),
            make_box("Answer"),
        ]

        flow = VGroup(*boxes).arrange(RIGHT, buff=0.8)

        arrows = VGroup(*[
            arrow(boxes[i], boxes[i + 1])
            for i in range(len(boxes) - 1)
        ])

        self.play(*[FadeIn(b) for b in boxes])
        self.play(*[GrowArrow(a) for a in arrows])

        self.play_voice("trad_query")
        self.show_sub("User question")

        self.play_voice("trad_embed")
        self.show_sub("Convert to embedding")

        self.play_voice("trad_search")
        self.show_sub("Vector search")

        self.play_voice("trad_llm")
        self.show_sub("Pass docs to LLM")

        self.play_voice("trad_answer")
        self.show_sub("Generate answer")

        self.wait(1)

        return VGroup(flow, arrows)

    # ─────────────────────────────────────────
    # Agentic RAG
    # ─────────────────────────────────────────

    def agentic(self):

        query = make_box("Query")
        agent = make_box("Agent")
        tools = make_box("Tools")
        vectordb = make_box("VectorDB")
        web = make_box("Web Search")
        context = make_box("Context")
        llm = make_box("LLM")
        answer = make_box("Answer")

        query.move_to(LEFT * 5)
        agent.next_to(query, RIGHT, buff=1)
        tools.next_to(agent, RIGHT, buff=1)

        vectordb.next_to(tools, UP * 2)
        web.next_to(tools, DOWN * 2)

        context.next_to(vectordb, RIGHT, buff=1)

        llm.next_to(context, DOWN, buff=1)

        answer.next_to(llm, LEFT, buff=1)

        arrows = VGroup(
            arrow(query, agent),
            arrow(agent, tools),
            Arrow(tools.get_right(), vectordb.get_left(), buff=0.25),
            Arrow(tools.get_right(), web.get_left(), buff=0.25),
            Arrow(vectordb.get_right(), context.get_left(), buff=0.25),
            Arrow(web.get_right(), context.get_left(), buff=0.25),
            v_arrow(context, llm),
            arrow(llm, answer)
        )

        loop = CurvedArrow(
            start_point=answer.get_left(),
            end_point=agent.get_right(),
            angle=-PI/2,
            color=WHITE
        )

        nodes = VGroup(
            query, agent, tools, vectordb, web, context, llm, answer
        )

        self.play(*[FadeIn(n) for n in nodes])
        self.play(*[GrowArrow(a) for a in arrows])
        self.play(Create(loop))

        self.play_voice("agent_reason")
        self.show_sub("Agent reasoning")

        self.play_voice("agent_tools")
        self.show_sub("Select tools")

        self.play_voice("agent_context")
        self.show_sub("Multiple retrieval")

        self.play_voice("agent_llm")
        self.show_sub("Provide context")

        self.play_voice("agent_answer")
        self.show_sub("Generate grounded answer")

        self.wait(2)

    # ─────────────────────────────────────────
    # Construct
    # ─────────────────────────────────────────

    def construct(self):

        self.intro()

        trad = self.traditional()

        self.play(trad.animate.shift(UP * 2))

        self.agentic()

        self.wait(3)