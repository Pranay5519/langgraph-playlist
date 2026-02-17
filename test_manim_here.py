from manim import *
import numpy as np

class NewtonScene(Scene):
    def construct(self):
        # 1. Title
        title = Text('Understanding the Sigmoid Function', font_size=40)
        # Positioned at y=+3.0
        title.move_to(UP * 3)

        # 2. Introduction text
        intro_text = Text(
            'The Sigmoid function, used in machine learning, maps any real number to a value between 0 and 1.',
            font_size=28
        )
        # Positioned relative to title with a vertical gap of 0.6 units
        intro_text.next_to(title, DOWN, buff=0.6)

        # 3. Coordinate system
        axes = Axes(
            x_range=[-6, 6, 1],
            y_range=[-3, 1, 0.5],
            axis_config={'color': WHITE}
        )
        # Axes are centered at (0,0) by default

        # 4. Sigmoid function plot
        sigmoid_graph = axes.plot(lambda x: 1 / (1 + np.exp(-x)), color=BLUE)

        # 5. Key Annotations
        # Following explicit coordinate requests, which overrides general stacking rule for these specific mobjects.
        annotation1 = Text('As x approaches -∞, f(x) approaches 0', font_size=24).move_to(np.array([-5, 0, 0]))
        annotation2 = Text('As x approaches +∞, f(x) approaches 1', font_size=24).move_to(np.array([5, 0, 0]))

        # 6. Animation
        self.play(
            Write(title),
            FadeIn(intro_text, shift=UP)
        )
        self.play(Create(axes), Create(sigmoid_graph))
        self.play(Write(annotation1), Write(annotation2))
        self.wait(2)