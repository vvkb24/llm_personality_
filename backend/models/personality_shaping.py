import json

class PersonalityShaper:
    def __init__(self):
        self.adjectives = {
            "EXT": {
                "high": ["talkative", "energetic", "bold", "active"],
                "low": ["silent", "timid", "inactive", "reserved"]
            }
        }

        self.intensity = {
            1: "extremely",
            2: "very",
            3: "",
            4: "a bit",
            5: "neither",
            6: "a bit",
            7: "",
            8: "very",
            9: "extremely"
        }

    def build_shaped_prompt(self, trait, level):
        if level < 5:
            adjectives = self.adjectives[trait]["low"]
        else:
            adjectives = self.adjectives[trait]["high"]

        prefix = self.intensity[level]
        phrase = ", ".join(f"{prefix} {adj}" for adj in adjectives)

        return f"Respond with the following personality: I am {phrase}."
