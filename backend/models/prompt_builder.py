class PromptBuilder:
    def build_prompt(self, persona, preamble, item, postamble):
        return f"""
For the following task, respond in a way that matches this description:
"{persona}"

{preamble} "{item['text']}", {postamble}
"""
