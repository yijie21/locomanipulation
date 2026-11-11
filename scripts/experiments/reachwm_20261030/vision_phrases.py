"""The phrase a person would ask for a SimReady object by, from its asset type (vision_grasp.py, vision_seg.py)."""

PHRASES = (  # the first key inside the asset's type names the object the way a person would ask for it
    ("abrasivePowder", "cleaning powder"), ("barSoap", "bar of soap"), ("brush", "brush"), ("bleach", "bleach bottle"),
    ("soapDispenser", "soap dispenser"), ("sponge", "sponge"), ("additive", "bottle"), ("antifreeze", "bottle"),
    ("grease", "tube"), ("sugarPacket", "sugar packet"), ("candle", "candle"), ("cup_paper", "paper cup"),
    ("cup_tea", "tea cup"), ("wineFlute", "champagne glass"), ("glass_wine", "wine glass"), ("glass", "glass"),
    ("mug", "mug"), ("plastic_cup", "plastic cup"), ("fork", "fork"), ("spoon_wooden", "wooden spoon"), ("spoon", "spoon"),
    ("boxedDrink", "juice box"), ("beverage_can", "can"), ("coffeeCup", "coffee cup"), ("ottle", "bottle"),
    ("food_can", "can"), ("ketchup", "ketchup bottle"), ("boxedCereal", "cereal box"), ("boxedFood", "box"),
    ("chocolate", "chocolate bar"), ("condiment", "bottle"), ("jam", "jar"), ("milk", "milk carton"), ("yogurt", "yogurt cup"),
    ("bellPepper", "bell pepper"), ("freshCarrot", "carrot"), ("sweetPotato", "sweet potato"), ("garlic", "garlic"),
    ("apple", "apple"), ("screwdriver", "screwdriver"), ("wrench", "wrench"), ("kitchenware_cup", "cup"), ("knife", "knife"),
    ("spatula", "spatula"), ("medicine", "medicine bottle"), ("chalk", "chalk"), ("marker", "marker"), ("pencil", "pencil"),
    ("pen", "pen"), ("pliers", "pliers"),
)  # fmt: skip


def phrase_of(kind: str) -> str:
    return next((p for key, p in PHRASES if key in kind), kind.split("_")[-1].lower())
