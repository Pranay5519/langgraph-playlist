from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService
import numpy as np

class NewtonScene(VoiceoverScene):
    def construct(self):
        self.set_service(GTTSService())
        
        # Tree and apple
        tree = ImageMobject("tree.png").scale(2)
        apple = Circle(color=RED, radius=0.2).shift(UP * 3 + RIGHT * 1.5)
        
        # Newton under the tree
        newton = ImageMobject("newton.png").scale(0.5).shift(DOWN * 2 + LEFT * 2)
        
        # Text elements
        earth_text = Text("Earth").shift(LEFT * 4)
        moon_text = Text("Moon").shift(RIGHT * 4)
        
        # Apple falling animation
        self.play(GrowFromCenter(tree))
        self.play(MoveAlongPath(apple, DOWN * 3 + LEFT * 1.5), run_time=2)
        
        # Newton's realization
        self.play(ShowCreation(newton))
        self.add_sound("apple_fall_sound.mp3")
        
        # Voiceover script
        self.voiceover(
            "Newton was inspired by an apple falling from a tree."
        )
        
        self.play(FadeIn(earth_text))
        self.play(FadeIn(moon_text))
        
        # Gravity concept
        gravity_arrow = Arrow(earth_text.get_center(), moon_text.get_center(), color=BLUE, buff=0.5)
        self.play(GrowArrow(gravity_arrow))
        
        self.voiceover(
            "This led him to realize gravity affects all objects in the universe."
        )
        
        # Final text
        law_text = Text("Law of Universal Gravitation").to_edge(UP)
        self.play(FadeIn(law_text))
        
        self.voiceover(
            "Newton's insight revolutionized our understanding of motion and gravity."
        )