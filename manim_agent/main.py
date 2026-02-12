from manim import *
import numpy as np

class LinearRegressionAnimation(Scene):
    def construct(self):

        # Subtitle
        subtitle = Text("Generating random data points", font_size=32).to_edge(DOWN)
        self.play(Write(subtitle))

        # Create invisible coordinate reference (just for positioning)
        origin = ORIGIN

        # Generate fake data
        np.random.seed(1)
        dots = VGroup()

        for i in range(-4, 5):
            x = i
            y = 0.5 * x + np.random.uniform(-1, 1)
            dot = Dot(point=np.array([x, y, 0]), color=BLUE)
            dots.add(dot)

        self.play(FadeIn(dots))
        self.wait(1)

        # Initial bad line
        new_subtitle = Text("Starting with a bad model", font_size=32).to_edge(DOWN)
        self.play(Transform(subtitle, new_subtitle))

        bad_line = Line(
            start=np.array([-5, 2, 0]),
            end=np.array([5, 2, 0]),
            color=RED
        )

        self.play(Create(bad_line))
        self.wait(1)

        # Better fitted line
        better_subtitle = Text("Improving the model to fit data", font_size=32).to_edge(DOWN)
        self.play(Transform(subtitle, better_subtitle))

        good_line = Line(
            start=np.array([-5, -2, 0]),
            end=np.array([5, 3, 0]),
            color=GREEN
        )

        self.play(Transform(bad_line, good_line))
        self.wait(2)
