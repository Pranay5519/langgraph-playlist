from manim import *

class OpenSourceLLMFamilyTree(Scene):
    def construct(self):

        title = Text("Open-Source LLM Model Families", font_size=40)
        title.to_edge(UP)

        def create_node(text, color=BLUE, width=3.5, height=0.9):
            box = RoundedRectangle(
                corner_radius=0.2,
                width=width,
                height=height,
                stroke_color=color
            )
            label = Text(text, font_size=20)
            label.move_to(box.get_center())
            return VGroup(box, label)

        # =============================
        # LLAMA FAMILY
        # =============================
        llama_root = create_node("Llama (Meta)", RED)

        llama_children = VGroup(
            create_node("Llama 2 (7B–70B)"),
            create_node("Llama 3 (8B, 70B)"),
            create_node("Llama 3.1 (8B–405B)"),
            create_node("Llama 3.2 (1B, 3B)"),
            create_node("Llama 4 (Multimodal)"),
            create_node("Code Llama"),
            create_node("Llama 3.2 Vision")
        ).arrange(DOWN, buff=0.35)

        llama_tree = VGroup(llama_root, llama_children).arrange(DOWN, buff=0.6)

        # =============================
        # MISTRAL FAMILY
        # =============================
        mistral_root = create_node("Mistral (Mistral AI)", GREEN)

        mistral_children = VGroup(
            create_node("Mistral 7B"),
            create_node("Mistral Small (24B)"),
            create_node("Mistral Large (123B)"),
            create_node("Mixtral (MoE)"),
            create_node("Codestral (22B)")
        ).arrange(DOWN, buff=0.35)

        mistral_tree = VGroup(mistral_root, mistral_children).arrange(DOWN, buff=0.6)

        # =============================
        # QWEN FAMILY
        # =============================
        qwen_root = create_node("Qwen (Alibaba)", ORANGE)

        qwen_children = VGroup(
            create_node("Qwen1.5"),
            create_node("Qwen2"),
            create_node("Qwen3 (MoE)"),
            create_node("Qwen2.5-Coder"),
            create_node("Qwen-VL"),
            create_node("Qwen2-Math")
        ).arrange(DOWN, buff=0.35)

        qwen_tree = VGroup(qwen_root, qwen_children).arrange(DOWN, buff=0.6)

        # =============================
        # PHI FAMILY
        # =============================
        phi_root = create_node("Phi (Microsoft)", PURPLE)

        phi_children = VGroup(
            create_node("Phi-3 (3.8B mini)"),
            create_node("Phi-4-mini-reasoning")
        ).arrange(DOWN, buff=0.35)

        phi_tree = VGroup(phi_root, phi_children).arrange(DOWN, buff=0.6)

        # =============================
        # GEMMA FAMILY
        # =============================
        gemma_root = create_node("Gemma (Google)", TEAL)

        gemma_children = VGroup(
            create_node("Gemma (2B, 7B)"),
            create_node("Gemma 2 (9B, 27B)"),
            create_node("Gemma 3"),
            create_node("CodeGemma"),
            create_node("EmbeddingGemma (300M)")
        ).arrange(DOWN, buff=0.35)

        gemma_tree = VGroup(gemma_root, gemma_children).arrange(DOWN, buff=0.6)

        # =============================
        # DEEPSEEK FAMILY
        # =============================
        deepseek_root = create_node("DeepSeek", YELLOW)

        deepseek_children = VGroup(
            create_node("DeepSeek-V3 (MoE)"),
            create_node("DeepSeek-V3.1"),
            create_node("DeepSeek-Coder"),
            create_node("DeepSeek-R1")
        ).arrange(DOWN, buff=0.35)

        deepseek_tree = VGroup(deepseek_root, deepseek_children).arrange(DOWN, buff=0.6)

        # =============================
        # ARRANGE ALL FAMILIES
        # =============================
        all_trees = VGroup(
            llama_tree,
            mistral_tree,
            qwen_tree,
            phi_tree,
            gemma_tree,
            deepseek_tree
        ).arrange(RIGHT, buff=1.0, aligned_edge=UP)

        all_trees.scale_to_fit_width(config.frame_width - 1)
        all_trees.next_to(title, DOWN, buff=0.5)

        self.add(title, all_trees)