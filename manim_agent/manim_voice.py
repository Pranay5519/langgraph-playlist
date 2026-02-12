from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

class LinearRegressionVoice(VoiceoverScene):
    def construct(self):

        # Set voice engine
        self.set_speech_service(GTTSService())

        # Voice + animation synced automatically
        with self.voiceover(text="Linear regression tries to fit a straight line to data points.") as tracker:
            title = Text("Linear Regression")
            self.play(Write(title), run_time=tracker.duration)

        self.wait()

        with self.voiceover(text="The model follows the equation y equals m x plus b."):
            equation = MathTex("y = mx + b").shift(DOWN)
            self.play(Write(equation))

        self.wait()
