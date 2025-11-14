class TestScorer:
    def score_likert(self, text):
        mapping = {
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5
        }
        return mapping.get(text.strip(), None)
