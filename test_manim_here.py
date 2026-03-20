from manim import *
import numpy as np

# ─── Loss surface (elongated bowl) ───────────────────────────────────
def loss(x, y):
    return 0.3 * x**2 + 2.5 * y**2

def grad(x, y):
    return np.array([0.6 * x, 5.0 * y])

def make_contours(ax, col1, col2):
    cg = VGroup()
    for level in [0.05, 0.2, 0.5, 1.0, 1.8, 2.8]:
        pts = []
        for angle in np.linspace(0, 2 * np.pi, 300):
            rx = np.sqrt(level / 0.3)
            ry = np.sqrt(level / 2.5)
            pts.append(ax.c2p(rx * np.cos(angle), ry * np.sin(angle)))
        cg.add(
            VMobject(
                stroke_color=interpolate_color(col1, col2, level / 3),
                stroke_width=1.5,
                fill_opacity=0,
            ).set_points_as_corners(pts + [pts[0]])
        )
    return cg


# ══════════════════════════════════════════════════════════════════════
# SCENE 1 — Introduction
# ══════════════════════════════════════════════════════════════════════
class IntroScene(Scene):
    def construct(self):
        title = Text("Gradient Descent Optimizers", font_size=52, color=WHITE)
        sub   = Text("AdaGrad  vs  Nesterov (NAG)", font_size=34, color=YELLOW)
        sub.next_to(title, DOWN, buff=0.45)
        desc  = Text(
            "Visualising how each optimizer navigates a loss landscape",
            font_size=24, color=GREY_A,
        ).next_to(sub, DOWN, buff=0.35)

        self.play(Write(title), run_time=1.4)
        self.play(FadeIn(sub, shift=UP * 0.3))
        self.play(FadeIn(desc, shift=UP * 0.2))
        self.wait(1.8)
        self.play(FadeOut(title), FadeOut(sub), FadeOut(desc))


# ══════════════════════════════════════════════════════════════════════
# SCENE 2 — AdaGrad
# ══════════════════════════════════════════════════════════════════════
class AdaGradScene(Scene):
    def construct(self):
        axes = Axes(
            x_range=[-3, 3, 1], y_range=[-2, 2, 1],
            x_length=10, y_length=6,
            axis_config={"color": GREY_B, "stroke_width": 2},
            tips=False,
        ).to_edge(DOWN, buff=0.4)
        x_lbl = axes.get_x_axis_label(Text("w₁", font_size=26))
        y_lbl = axes.get_y_axis_label(Text("w₂", font_size=26))

        title = Text("AdaGrad", font_size=42, color=ORANGE).to_edge(UP, buff=0.25)
        formula = MathTex(
            r"G_t = G_{t-1} + g_t^2,\quad"
            r"\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \varepsilon}}\,g_t",
            font_size=28, color=YELLOW,
        ).next_to(title, DOWN, buff=0.18)

        self.play(Write(title), run_time=0.7)
        self.play(Write(formula), run_time=1.1)
        self.play(Create(axes), Write(x_lbl), Write(y_lbl), run_time=0.9)
        self.play(Create(make_contours(axes, BLUE_E, TEAL)), run_time=1.4)

        # ── Compute AdaGrad path ──────────────────────────────────────
        lr, eps = 0.9, 1e-8
        th = np.array([2.5, 1.5])
        G  = np.zeros(2)
        path = [th.copy()]
        for _ in range(60):
            g  = grad(*th)
            G += g ** 2
            th = th - lr / (np.sqrt(G) + eps) * g
            path.append(th.copy())

        dot   = Dot(axes.c2p(*path[0]), color=ORANGE, radius=0.12)
        trail = VMobject(stroke_color=ORANGE, stroke_width=3, fill_opacity=0)
        trail.set_points_as_corners([axes.c2p(*path[0])] * 2)
        self.play(FadeIn(dot))

        note = Text("Learning rate shrinks over time", font_size=21, color=ORANGE)
        note.to_corner(DL, buff=0.3)
        self.play(FadeIn(note, shift=RIGHT * 0.2))

        prev = axes.c2p(*path[0])
        for i, pt in enumerate(path[1:]):
            nxt = axes.c2p(*pt)
            arr = Arrow(
                prev, nxt, buff=0, stroke_width=2.5,
                color=interpolate_color(ORANGE, RED_D, i / len(path)),
                max_tip_length_to_length_ratio=0.28,
            )
            new_trail = trail.copy()
            new_trail.add_points_as_corners([nxt])
            self.play(
                Create(arr),
                Transform(trail, new_trail),
                dot.animate.move_to(nxt),
                run_time=0.07,
            )
            prev = nxt

        conv = Text("✓ Converged", font_size=26, color=GREEN).next_to(dot, UR, buff=0.15)
        self.play(FadeIn(conv, scale=1.3))
        self.wait(1.2)

        btext = VGroup(
            Text("AdaGrad", font_size=22, color=ORANGE, weight=BOLD),
            Text("✚ Per-parameter adaptive lr", font_size=19, color=GREEN_B),
            Text("✚ Great for sparse gradients", font_size=19, color=GREEN_B),
            Text("✖ lr monotonically decays → stalls", font_size=19, color=RED_B),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        box = SurroundingRectangle(btext, color=ORANGE, buff=0.18)
        VGroup(btext, box).to_corner(UR, buff=0.25)
        self.play(Create(box), Write(btext), run_time=1.4)
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ══════════════════════════════════════════════════════════════════════
# SCENE 3 — Nesterov Accelerated Gradient
# ══════════════════════════════════════════════════════════════════════
class NesterovScene(Scene):
    def construct(self):
        axes = Axes(
            x_range=[-3, 3, 1], y_range=[-2, 2, 1],
            x_length=10, y_length=6,
            axis_config={"color": GREY_B, "stroke_width": 2},
            tips=False,
        ).to_edge(DOWN, buff=0.4)
        x_lbl = axes.get_x_axis_label(Text("w₁", font_size=26))
        y_lbl = axes.get_y_axis_label(Text("w₂", font_size=26))

        title = Text(
            "Nesterov Accelerated Gradient (NAG)", font_size=36, color=TEAL_A 
        ).to_edge(UP, buff=0.25)
        formula = MathTex(
            r"v_{t+1} = \mu v_t - \eta\,\nabla f(\theta_t + \mu v_t),\quad"
            r"\theta_{t+1} = \theta_t + v_{t+1}",
            font_size=26, color=YELLOW,
        ).next_to(title, DOWN, buff=0.18)

        self.play(Write(title), run_time=0.7)
        self.play(Write(formula), run_time=1.1)
        self.play(Create(axes), Write(x_lbl), Write(y_lbl), run_time=0.9)
        self.play(Create(make_contours(axes, BLUE_E, PURPLE_B)), run_time=1.4)

        # ── Compute NAG path ──────────────────────────────────────────
        lr, mu = 0.08, 0.9
        th = np.array([2.5, 1.5])
        v  = np.zeros(2)
        path, la_pts = [th.copy()], []
        for _ in range(55):
            la = th + mu * v
            la_pts.append(la.copy())
            g  = grad(*la)
            v  = mu * v - lr * g
            th = th + v
            path.append(th.copy())

        dot   = Dot(axes.c2p(*path[0]), color=TEAL_A , radius=0.12)
        trail = VMobject(stroke_color=TEAL_A , stroke_width=3, fill_opacity=0)
        trail.set_points_as_corners([axes.c2p(*path[0])] * 2)
        self.play(FadeIn(dot))

        note = Text(
            "Evaluates gradient at look-ahead point (dashed)",
            font_size=21, color=YELLOW,
        ).to_corner(DL, buff=0.3)
        self.play(FadeIn(note, shift=RIGHT * 0.2))

        prev = axes.c2p(*path[0])
        for i, (pt, la) in enumerate(zip(path[1:], la_pts)):
            nxt  = axes.c2p(*pt)
            la_p = axes.c2p(*la)

            la_arr = DashedVMobject(
                Arrow(prev, la_p, buff=0, stroke_width=2,
                      color=YELLOW, max_tip_length_to_length_ratio=0.2),
                num_dashes=8,
            )
            step_arr = Arrow(
                prev, nxt, buff=0, stroke_width=3,
                color=interpolate_color(TEAL_A , BLUE, i / len(path)),
                max_tip_length_to_length_ratio=0.28,
            )
            new_trail = trail.copy()
            new_trail.add_points_as_corners([nxt])

            self.play(Create(la_arr), run_time=0.04)
            self.play(
                FadeOut(la_arr),
                Create(step_arr),
                Transform(trail, new_trail),
                dot.animate.move_to(nxt),
                run_time=0.07,
            )
            prev = nxt

        conv = Text("✓ Converged (faster!)", font_size=26, color=GREEN).next_to(dot, UR, buff=0.15)
        self.play(FadeIn(conv, scale=1.3))
        self.wait(1.2)

        btext = VGroup(
            Text("NAG", font_size=22, color=TEAL_A , weight=BOLD),
            Text("✚ Anticipates next position", font_size=19, color=GREEN_B),
            Text("✚ Less oscillation than momentum SGD", font_size=19, color=GREEN_B),
            Text("✚ Faster convergence", font_size=19, color=GREEN_B),
            Text("✖ Needs careful lr / μ tuning", font_size=19, color=RED_B),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        box = SurroundingRectangle(btext, color=TEAL_A , buff=0.18)
        VGroup(btext, box).to_corner(UR, buff=0.25)
        self.play(Create(box), Write(btext), run_time=1.4)
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ══════════════════════════════════════════════════════════════════════
# SCENE 4 — Side-by-side comparison + loss curve
# ══════════════════════════════════════════════════════════════════════
class ComparisonScene(Scene):
    def construct(self):
        title = Text("Side-by-Side Comparison", font_size=38, color=WHITE).to_edge(UP, buff=0.25)
        self.play(Write(title))

        axL = Axes(
            x_range=[-3, 3, 1], y_range=[-2, 2, 1],
            x_length=5.2, y_length=3.4,
            axis_config={"color": GREY_B}, tips=False,
        ).shift(LEFT * 3.1 + DOWN * 0.4)
        axR = Axes(
            x_range=[-3, 3, 1], y_range=[-2, 2, 1],
            x_length=5.2, y_length=3.4,
            axis_config={"color": GREY_B}, tips=False,
        ).shift(RIGHT * 3.1 + DOWN * 0.4)

        labL = Text("AdaGrad", font_size=28, color=ORANGE).next_to(axL, UP, buff=0.08)
        labR = Text("NAG",     font_size=28, color=TEAL_A  ).next_to(axR, UP, buff=0.08)

        self.play(
            Create(axL), Create(axR),
            Write(labL), Write(labR),
            Create(make_contours(axL, BLUE_E, TEAL)),
            Create(make_contours(axR, BLUE_E, PURPLE_B)),
            run_time=1.5,
        )

        # ── Paths ─────────────────────────────────────────────────────
        lr, eps = 0.9, 1e-8
        thA = np.array([2.5, 1.4])
        GA  = np.zeros(2)
        ada_path = [thA.copy()]
        for _ in range(50):
            g = grad(*thA); GA += g ** 2
            thA = thA - lr / (np.sqrt(GA) + eps) * g
            ada_path.append(thA.copy())

        lr2, mu = 0.08, 0.9
        thN = np.array([2.5, 1.4])
        vN  = np.zeros(2)
        nag_path = [thN.copy()]
        for _ in range(50):
            g = grad(*(thN + mu * vN)); vN = mu * vN - lr2 * g
            thN = thN + vN
            nag_path.append(thN.copy())

        dL = Dot(axL.c2p(*ada_path[0]), color=ORANGE, radius=0.10)
        dR = Dot(axR.c2p(*nag_path[0]), color=TEAL_A ,   radius=0.10)
        trL = VMobject(stroke_color=ORANGE, stroke_width=2.5, fill_opacity=0)
        trR = VMobject(stroke_color=TEAL_A ,   stroke_width=2.5, fill_opacity=0)
        trL.set_points_as_corners([axL.c2p(*ada_path[0])] * 2)
        trR.set_points_as_corners([axR.c2p(*nag_path[0])] * 2)
        self.play(FadeIn(dL), FadeIn(dR))

        for i in range(1, 51):
            nL = axL.c2p(*ada_path[i]); nR = axR.c2p(*nag_path[i])
            newL = trL.copy(); newL.add_points_as_corners([nL])
            newR = trR.copy(); newR.add_points_as_corners([nR])
            self.play(
                Transform(trL, newL), dL.animate.move_to(nL),
                Transform(trR, newR), dR.animate.move_to(nR),
                run_time=0.09,
            )

        # ── Loss-curve chart ──────────────────────────────────────────
        losses_ada = [loss(*p) for p in ada_path]
        losses_nag = [loss(*p) for p in nag_path]

        chart_axes = Axes(
            x_range=[0, 50, 10], y_range=[0, 6, 2],
            x_length=9, y_length=2,
            axis_config={"color": GREY_B, "stroke_width": 1.5},
            tips=False,
        ).to_edge(DOWN, buff=0.15)
        ca_lbl = Text("Loss over iterations", font_size=18, color=GREY_A
                      ).next_to(chart_axes, UP, buff=0.05)
        self.play(Create(chart_axes), Write(ca_lbl), run_time=0.7)

        ada_line = chart_axes.plot_line_graph(
            x_values=list(range(51)), y_values=losses_ada,
            line_color=ORANGE, stroke_width=2, add_vertex_dots=False,
        )
        nag_line = chart_axes.plot_line_graph(
            x_values=list(range(51)), y_values=losses_nag,
            line_color=TEAL_A , stroke_width=2, add_vertex_dots=False,
        )

        leg_a = VGroup(
            Line(ORIGIN, RIGHT * 0.5, color=ORANGE, stroke_width=3),
            Text("AdaGrad", font_size=16, color=ORANGE),
        ).arrange(RIGHT, buff=0.1)
        leg_n = VGroup(
            Line(ORIGIN, RIGHT * 0.5, color=TEAL_A , stroke_width=3),
            Text("NAG", font_size=16, color=TEAL_A ),
        ).arrange(RIGHT, buff=0.1)
        leg = VGroup(leg_a, leg_n).arrange(RIGHT, buff=0.4).next_to(ca_lbl, RIGHT, buff=0.4)

        self.play(Create(ada_line), Create(nag_line), Write(leg), run_time=1.8)
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects])

        outro = Text(
            "Both optimizers reach the minimum —\nbut via different strategies.",
            font_size=32, color=WHITE, line_spacing=1.4,
        )
        self.play(Write(outro))
        self.wait(2)
        self.play(FadeOut(outro))


# ══════════════════════════════════════════════════════════════════════
# Combined entry-point  (run: manim -pqh optimizer_viz.py OptimizerVisualization)
# ══════════════════════════════════════════════════════════════════════
class OptimizerVisualization(Scene):
    def construct(self):
        for Cls in [IntroScene, AdaGradScene, NesterovScene, ComparisonScene]:
            Cls.construct(self)