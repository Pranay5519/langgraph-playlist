from manim import *
import os
from dotenv import load_dotenv
from elevenlabs import ElevenLabs, save

class RadiusWithVoice(Scene):
    def construct(self):

        # ---------------------------
        # Generate Voice using ElevenLabs
        # ---------------------------
        load_dotenv()
        client = ElevenLabs(api_key=os.getenv("ELEVEN_API_KEY"))

        narration_text = (
            "This is a circle. "
            "The radius is the distance from the center of the circle to its boundary."
        )

        audio = client.text_to_speech.convert(
            voice_id="21m00Tcm4TlvDq8ikWAM",
            model_id="eleven_multilingual_v2",
            text=narration_text
        )

        save(audio, "radius_voice.mp3")

        # ---------------------------
        # Visual Part
        # ---------------------------

        circle = Circle(radius=2, color=BLUE)
        center = circle.get_center()

        center_dot = Dot(center, color=RED)
        center_label = Text("Center").scale(0.5).next_to(center_dot, DOWN)

        radius_line = Line(center, circle.point_at_angle(0), color=YELLOW)
        radius_label = Text("Radius (r)").scale(0.6).next_to(radius_line, UP)

        subtitle = Text(
            "Radius = distance from center to boundary",
            font_size=28
        ).to_edge(DOWN)

        # ---------------------------
        # Animation + Voice
        # ---------------------------

        self.play(Create(circle))
        self.play(FadeIn(center_dot), Write(center_label))
        self.play(Create(radius_line), Write(radius_label))

        self.add_sound("radius_voice.mp3")
        self.play(Write(subtitle), run_time=6)

        self.wait(2)