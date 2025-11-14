from fastapi import FastAPI
from backend.models.llm_engine import LLMEngine
from backend.models.prompt_builder import PromptBuilder
from backend.models.psychometrics import PsychometricEngine
from backend.models.personality_shaping import PersonalityShaper
from backend.models.scoring import TestScorer
from backend.routers.text_api import router as text_router
import json

app = FastAPI()

# Initialize components
llm = LLMEngine()
prompt_builder = PromptBuilder()
psych_engine = PsychometricEngine()
shaper = PersonalityShaper()
scorer = TestScorer()

# include text API router
app.include_router(text_router, prefix="/api")


@app.get("/")
def root():
	return {"message": "LLM personality system running"}


@app.post("/measure/ipip")
def measure_ipip(payload: dict = None):
	items = json.load(open("data/ipip_neo_300.json"))
	personas = json.load(open("data/personas.json"))
	preambles = json.load(open("data/item_preambles.json"))
	postambles = json.load(open("data/item_postambles.json"))

	all_scores = []

	for persona in personas:
		for pre in preambles:
			for post in postambles:
				for item in items:
					prompt = prompt_builder.build_prompt(persona, pre, item, post)
					score = llm.score_prompt(prompt)

					all_scores.append({
						"persona": persona,
						"item_id": item["id"],
						"domain": item["domain"],
						"facet": item.get("facet", None),
						"score": score
					})

	stats = psych_engine.compute_validity(all_scores)

	return {"scores": all_scores, "validity": stats}


@app.post("/shape")
def shape_personality(payload: dict):
	trait = payload["trait"]
	level = payload["level"]
	text = payload.get("text", "")

	shaped_prompt = shaper.build_shaped_prompt(trait, level)
	final_text = llm.generate(shaped_prompt + "\n" + text)

	return {"prompt": shaped_prompt, "output": final_text}
