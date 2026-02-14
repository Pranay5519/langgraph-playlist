from manim import *
from gtts import gTTS
import os
import numpy as np
from manim_voiceover import VoiceoverScene

class DeepLearningWorkingScene(VoiceoverScene):
    def construct(self):
        # Axes setup
        input_axes = Axes(
            x_range=[-2, 2], y_range=[-1, 1],
            axis_config={
                "include_numbers": False,
                "label_direction": RIGHT
            }
        ).shift(LEFT*4)
        output_axes = Axes(
            x_range=[-2, 2], y_range=[-1, 1],
            axis_config={
                "include_numbers": False,
                "label_direction": LEFT
            }
        ).shift(RIGHT*4)

        input_label = Text("Input").next_to(input_axes, DOWN)
        output_label = Text("Output").next_to(output_axes, UP)

        # Neuron layers
        input_neurons = VGroup(*[
            Circle(radius=0.2).move_to(input_axes.c2p(i, 0))
            for i in range(3)
        ])
        output_neurons = VGroup(*[
            Circle(radius=0.2).move_to(output_axes.c2p(i, 0))
            for i in range(3)
        ])

        # Connection weights
        connections = VGroup()
        for i in range(3):
            for j in range(3):
                arrow = Arrow(
                    input_neurons[i].get_right(),
                    output_neurons[j].get_left(),
                    buff=0.1
                ).set_color(GREY)
                connections.add(arrow)

        # Activation functions
        activation_labels = VGroup()
        for neuron in output_neurons:
            activation_label = Text("σ").next_to(neuron, RIGHT)
            activation_labels.add(activation_label)

        # Backpropagation gradients
        gradient_arrows = VGroup()
        for i in range(3):
            gradient_arrow = Arrow(
                output_neurons[i].get_right(),
                input_neurons[i].get_left(),
                buff=0.1,
                color=YELLOW
            ).set_opacity(0.5)
            gradient_arrows.add(gradient_arrow)

        # Voiceover blocks
        self.add(input_axes, output_axes, input_label, output_label)

        # Step 1: Introduce neural network layers
        self.play(
            Create(input_neurons),
            Create(output_neurons),
            run_time=3
        )
        self.add_sound("voiceover1.mp3")
        self.wait(2)

        # Step 2: Show weighted connections
        self.play(
            Create(connections),
            run_time=3
        )
        self.add_sound("voiceover2.mp3")
        self.wait(2)

        # Step 3: Forward propagation with activation
        self.play(
            Write(activation_labels),
            run_time=2
        )
        self.add_sound("voiceover3.mp3")
        self.wait(2)

        # Step 4: Backpropagation with gradients
        self.play(
            Create(gradient_arrows),
            run_time=3
        )
        self.add_sound("voiceover4.mp3")
        self.wait(2)

        # Generate and play GTTS voiceovers
        def generate_audio(text, filename):
            tts = gTTS(text=text, lang='en')
            tts.save(filename)
            os.system(f"mpg123 {filename}")

        generate_audio("This is the first voiceover.", "voiceover1.mp3")
        generate_audio("This is the second voiceover.", "voiceover2.mp3")
        generate_audio("This is the third voiceover.", "voiceover3.mp3")
        generate_audio("This is the fourth voiceover.", "voiceover4.mp3")