#!/usr/bin/env python3
"""Generate creative writing prompts and save them to a file.

This script generates prompts once so they can be reused across multiple model evaluations.
"""
import argparse
import json
import random


CREATIVE_WRITING_PROMPTS = [
    "Write a short story about a detective who discovers that their reflection has been living a separate life.",
    "Describe a sunset from the perspective of someone seeing color for the first time.",
    "Write a dialogue between the moon and the ocean during a particularly high tide.",
    "Compose a letter from a tree to the person who planted it 50 years ago.",
    "Write a scene where a chef tastes a dish that transports them back to their childhood.",
    "Describe a bustling marketplace in a city that exists between dimensions.",
    "Write about a musician who finds an instrument that plays people's memories instead of notes.",
    "Create a story about the last bookstore on Earth and its peculiar owner.",
    "Write a poem about the journey of a raindrop from cloud to ocean.",
    "Describe a conversation between two strangers waiting for a train that never comes.",
    "Write about a painter who discovers their paintings can alter reality.",
    "Compose a monologue from the perspective of an ancient statue watching the city change around it.",
    "Write a story about a librarian who can read the unwritten stories in people's hearts.",
    "Describe the first meal shared between two rival chefs forced to work together.",
    "Write about a gardener who grows emotions instead of flowers.",
    "Create a scene in a café where time moves differently for each customer.",
    "Write a letter from the future to the present, warning about something beautiful that will be lost.",
    "Describe a world where music is visible and colors are audible.",
    "Write about a clockmaker who creates timepieces that show alternate timelines.",
    "Compose a story about the friendship between a lighthouse keeper and a sea creature.",
    "Write a scene where someone discovers a door in their house that wasn't there yesterday.",
    "Describe a festival celebrating the return of forgotten words to the language.",
    "Write about a photographer who captures souls instead of images.",
    "Create a dialogue between a sword and the warrior who wields it.",
    "Write a story about a baker whose bread can heal emotional wounds.",
    "Describe the last conversation between the sun and the moon before an eternal eclipse.",
    "Write about a dancer whose movements can change the weather.",
    "Compose a scene in a museum where the paintings gossip about the visitors.",
    "Write a story about a mapmaker who charts emotional landscapes.",
    "Describe a library where books choose their readers instead of the other way around.",
    "Write about a tailor who sews dreams into clothing.",
    "Create a monologue from a bridge that has witnessed a century of crossings.",
    "Write a scene where a perfumer creates a scent that captures a specific moment in time.",
    "Describe a garden that blooms only under starlight.",
    "Write about a street performer who can make people's shadows dance independently.",
    "Compose a story about the keeper of a lighthouse that guides ships through time, not space.",
    "Write a dialogue between a pen and the blank page it's about to mark.",
    "Describe a city where buildings are grown from seeds instead of constructed.",
    "Write about a watchmaker who repairs broken hearts instead of broken clocks.",
    "Create a scene in a restaurant where each dish serves a memory instead of food.",
    "Write a story about a weaver who creates tapestries that show possible futures.",
    "Describe the meeting between winter and spring at the exact moment of seasonal change.",
    "Write about a singer whose voice can make flowers bloom or wither.",
    "Compose a letter from the ocean to the rivers that feed it.",
    "Write a scene where a sculptor discovers their clay figures come alive at midnight.",
    "Describe a bookshop where the books rearrange themselves based on what readers need, not want.",
    "Write about a glassblower who captures lightning in bottles.",
    "Create a story about a cartographer mapping the geography of dreams.",
    "Write a dialogue between a candle and the darkness it holds at bay.",
    "Describe a concert hall where the acoustics reveal truths the musicians didn't intend to share.",
    "Write about a jeweler who crafts accessories from crystallized laughter and tears.",
    "Compose a scene in a theater where the audience becomes part of the play.",
    "Write a story about a florist who arranges bouquets that bloom into specific emotions.",
    "Describe the perspective of a compass that always points toward what someone needs most.",
    "Write about a potter whose vessels hold not water, but whispered secrets.",
    "Create a monologue from a fountain that has granted wishes for centuries.",
    "Write a scene where a calligrapher's words float off the page and into reality.",
    "Describe a marketplace where merchants trade in moments rather than money.",
    "Write about a blacksmith who forges keys that unlock memories, not doors.",
    "Compose a story about a shepherd who tends to clouds instead of sheep.",
    "Write a dialogue between a mirror and the person who refuses to look into it.",
    "Describe a festival where people exchange their shadows for a day.",
    "Write about a vintner whose wines contain the essence of specific seasons.",
    "Create a scene in a gallery where portraits age while their subjects remain young.",
    "Write a story about a bell-ringer whose chimes can summon specific emotions.",
    "Describe the journey of a message in a bottle that travels through time, not water.",
    "Write about a seamstress who mends relationships along with clothing.",
    "Compose a letter from autumn to summer, explaining why change is necessary.",
    "Write a scene where a chef prepares a meal using ingredients from different eras.",
    "Describe a garden where each plant grows from a planted wish.",
    "Write about a musician who collects silence in jars.",
    "Create a story about a lamplighter who illuminates not streets, but forgotten memories.",
    "Write a dialogue between a key and the lock it was made for but has never met.",
    "Describe a workshop where toys are repaired and given new purposes.",
    "Write about a storyteller whose tales become real for those who truly believe.",
    "Compose a scene in a train station where each platform leads to a different version of the same destination.",
    "Write a story about a beekeeper whose bees collect dreams instead of pollen.",
    "Describe the conversation between a shooting star and the wish made upon it.",
    "Write about a cobbler who makes shoes that let people walk through their fears.",
    "Create a monologue from a door that has opened to countless lives but never moved.",
    "Write a scene where a tea master brews beverages that reveal hidden truths.",
    "Describe a forest where trees grow from planted stories instead of seeds.",
    "Write about a kite maker whose creations carry messages to the departed.",
    "Compose a story about a night watchman who guards the boundary between dreams and reality.",
    "Write a dialogue between the first and last pages of an unfinished book.",
    "Describe a bridge that connects not two shores, but two states of mind.",
    "Write about a mask maker whose creations reveal true faces rather than hide them.",
    "Create a scene in a workshop where broken instruments are taught to play new songs.",
    "Write a story about a ferryman who transports people between their past and future selves.",
    "Describe the moment when a blank canvas decides what it wants to become.",
    "Write about a locksmith who creates keys to unlock potential, not doors.",
    "Compose a letter from the stars to a child who wishes upon them every night.",
    "Write a scene where a translator discovers they can interpret the language of colors.",
    "Describe a harbor where ships dock carrying cargo from parallel universes.",
    "Write about a quilter who stitches together fragments of forgotten songs.",
    "Create a story about a guide who leads tours through people's memories.",
    "Write a dialogue between dawn and dusk about the day that passed between them.",
    "Describe a studio where portraits are painted not of faces, but of souls.",
    "Write about a herbalist who grows remedies for ailments that don't exist yet.",
    "Compose a scene in a post office where letters are sorted by emotional weight, not address.",
]


def main():
    parser = argparse.ArgumentParser(
        description="Generate creative writing prompts for model evaluation"
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=100,
        help="Number of prompts to generate (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="creative_writing_prompts.jsonl",
        help="Output file (default: creative_writing_prompts.jsonl)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for prompt selection (default: 42)",
    )
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    if args.num_prompts <= len(CREATIVE_WRITING_PROMPTS):
        selected_prompts = random.sample(CREATIVE_WRITING_PROMPTS, args.num_prompts)
    else:
        selected_prompts = random.choices(CREATIVE_WRITING_PROMPTS, k=args.num_prompts)
    
    with open(args.output, "w") as f:
        for i, prompt in enumerate(selected_prompts):
            f.write(json.dumps({"id": i, "prompt": prompt}, ensure_ascii=False) + "\n")
    
    print(f"Generated {len(selected_prompts)} prompts")
    print(f"Saved to: {args.output}")
    print(f"Seed: {args.seed}")


if __name__ == "__main__":
    main()
