# -*- coding: utf-8 -*-
"""
AI NEGOTIATION AGENT - Interactive Chat with LLaMA 3.1:8B
Save as negotiation_agent_manoj_ollama_summary.py and run with: python negotiation_agent_manoj_ollama_summary.py
"""

import sys
import re
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import subprocess

OLLAMA_MODEL = "llama3.1:8b"  # ensure installed locally
MAX_ROUNDS = 10  # max negotiation rounds

# ============================================
# DATA STRUCTURES
# ============================================

@dataclass
class Product:
    name: str
    category: str
    quantity: int
    quality_grade: str
    origin: str
    base_market_price: int
    attributes: Dict[str, Any]

@dataclass
class NegotiationContext:
    product: Product
    your_budget: int
    current_round: int
    seller_offers: List[int]
    your_offers: List[int]
    messages: List[Dict[str, str]]

class DealStatus(Enum):
    ONGOING = "ongoing"
    ACCEPTED = "accepted"
    REJECTED = "rejected"

# ============================================
# BASE AGENT
# ============================================

class BaseBuyerAgent(ABC):
    def __init__(self, name: str):
        self.name = name
        self.personality = self.define_personality()

    @abstractmethod
    def define_personality(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def respond_to_seller_offer(self, context: NegotiationContext, seller_price: int, seller_message: str) -> Tuple[DealStatus, int, str]:
        pass

    @abstractmethod
    def get_personality_prompt(self) -> str:
        pass

# ============================================
# DIPLOMATIC BUYER AGENT
# ============================================

class YourBuyerAgent(BaseBuyerAgent):

    def define_personality(self) -> Dict[str, Any]:
        return {
            "personality_type": "assertive_value_protector",
            "traits": ["confident", "persuasive", "value-conscious", "strategic"],
            "catchphrases": [
                "Given the quality here, my offer is already fair.",
                "This price reflects both market reality and product value."
            ]
        }

    def respond_to_seller_offer(
        self,
        context: NegotiationContext,
        seller_price: int,
        seller_message: str
    ) -> Tuple[DealStatus, int, str]:
        fair_price = self.calculate_fair_price(context.product)
        last_offer = context.your_offers[-1] if context.your_offers else int(fair_price * 0.75)
        max_willing = context.your_budget
        min_willing = max(1, int(fair_price * 0.6))

        # --- Round-specific logic ---
        if context.current_round == 9:
            # Ask for 10% reduction
            target_price = int(seller_price * 0.9)
            if target_price > max_willing:
                target_price = max_willing
            if target_price < min_willing:
                target_price = min_willing
            msg = f"If you can reduce by 10% to ₹{target_price}, we have a deal."
            return DealStatus.ONGOING, target_price, msg

        if context.current_round == 10:
            # Accept regardless
            return DealStatus.ACCEPTED, seller_price, f"Alright, I accept ₹{seller_price}."

        # --- Normal negotiation logic ---
        prompt = f"""
You are a confident, value-protecting buyer negotiating for {context.product.quantity} x {context.product.name} (quality: {context.product.quality_grade}).
Market price: ₹{context.product.base_market_price}, Fair price: ₹{fair_price}, Buyer budget: ₹{context.your_budget}.
Seller offer: ₹{seller_price} — "{seller_message}".
Your last offer: ₹{last_offer}.
Your goal: Protect your budget, aim for best deal without going above ₹{max_willing}.
Respond firmly with either 'ACCEPT' or a confident counteroffer (must not exceed ₹{max_willing}, and not go below ₹{min_willing}).
Keep messages 1–2 sentences, persuasive, matching your final numeric offer exactly.
"""
        ai_reply = self.query_ollama(prompt, fallback_price=last_offer)
        counter = self.extract_price(ai_reply)

        if counter > max_willing:
            counter = max_willing
            ai_reply = f"Considering my budget and the value, I can only go up to ₹{counter}."
        elif counter < min_willing:
            counter = min_willing
            ai_reply = f"This is my best and final offer given the market reality — ₹{counter}."

        tolerance = int(fair_price * 0.02)
        if seller_price <= max_willing and seller_price <= fair_price + tolerance:
            return DealStatus.ACCEPTED, seller_price, f"Alright, I accept ₹{seller_price}."

        return DealStatus.ONGOING, counter, ai_reply

    def query_ollama(self, prompt: str, fallback_price: int) -> str:
        try:
            result = subprocess.run(
                ["ollama", "run", OLLAMA_MODEL],
                input=prompt.encode("utf-8"),
                capture_output=True,
                check=True
            )
            response = result.stdout.decode("utf-8").strip()
            return response if response else f"I can offer ₹{fallback_price}."
        except Exception:
            return f"I can offer ₹{fallback_price}."

    def extract_price(self, text: str) -> int:
        s = text.replace(",", "").replace("₹", "").strip()
        match = re.search(r"(\d+(\.\d+)?)", s)
        return int(float(match.group(1))) if match else 0

    def calculate_fair_price(self, product: Product) -> int:
        base = product.base_market_price
        grade_adj = {"A": 1.05, "B": 0.95, "Export": 1.10}
        multiplier = grade_adj.get(product.quality_grade, 1.0)
        if product.attributes.get("export_grade"):
            multiplier += 0.02
        fair = int(base * multiplier)
        fair = max(int(base * 0.7), min(int(base * 1.2), fair))
        return fair

    def get_personality_prompt(self) -> str:
        return "You are a confident, firm, value-protecting buyer. Persuasive but concise."


# ============================================
# NEGOTIATION SUMMARY
# ============================================

def summarize_negotiation(context: NegotiationContext, final_price: int):
    fair_price = YourBuyerAgent(name="Temp").calculate_fair_price(context.product)
    print("\n=== NEGOTIATION SUMMARY ===")
    print(f"Product: {context.product.quantity} x {context.product.name} (quality: {context.product.quality_grade})")
    print(f"Buyer Budget: ₹{context.your_budget}")
    print(f"Calculated Fair Price: ₹{fair_price}")
    print(f"Final Price Agreed: ₹{final_price}")
    print(f"Total Rounds: {context.current_round}\n")

    buyer_savings = context.your_budget - final_price
    below_market = context.product.base_market_price - final_price

    print(f"Buyer Savings: ₹{buyer_savings} ({buyer_savings/context.your_budget*100:.1f}%)")
    print(f"Below Market Price: ₹{below_market} ({below_market/context.product.base_market_price*100:.1f}%)")

    if final_price < fair_price:
        winner = "Buyer won — got a price below fair value!"
    elif final_price > fair_price:
        winner = "Seller won — sold above fair value!"
    else:
        winner = "Balanced deal — price near fair value."
    print(f"Result: {winner}\n")

    print("=== Full Conversation ===")
    for msg in context.messages:
        role = "Buyer" if msg["role"]=="buyer" else "Seller"
        print(f"{role}: {msg['message']}")

# ============================================
# INTERACTIVE CHAT
# ============================================

def chat_with_llama():
    print("=== Chat with YourBuyerAgent (LLaMA 3.1:8B) ===")
    print(f"Maximum Rounds: {MAX_ROUNDS}")
    print("Type your message as seller (include price if you want). Type 'exit' to quit.\n")

    agent = YourBuyerAgent(name="ChatBuyer")

    product = Product(
        name="Alphonso Mangoes",
        category="Fruit",
        origin="India",
        quantity=10,
        base_market_price=500,
        quality_grade="A",
        attributes={"export_grade": True}
    )

    context = NegotiationContext(
        product=product,
        your_budget=450,
        current_round=0,
        seller_offers=[],
        your_offers=[],
        messages=[]
    )

    while context.current_round < MAX_ROUNDS:
        user_input = input("You (Seller): ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chat...")
            break

        match = re.search(r"(\d+)", user_input.replace(",", ""))
        seller_price = int(match.group(1)) if match else product.base_market_price
        seller_message = user_input

        context.current_round += 1
        context.seller_offers.append(seller_price)
        context.messages.append({"role": "seller", "message": seller_message})

        status, ai_offer, ai_msg = agent.respond_to_seller_offer(context, seller_price, seller_message)
        context.your_offers.append(ai_offer)
        context.messages.append({"role": "buyer", "message": ai_msg})

        print(f"AI Buyer: ₹{ai_offer} — {ai_msg}")

        if status == DealStatus.ACCEPTED:
            print(f"\n--- DEAL MADE at ₹{ai_offer}! ---")
            summarize_negotiation(context, ai_offer)
            break
    else:
        # Max rounds reached without deal
        print("\n--- MAX ROUNDS REACHED — Negotiation ended without a deal. ---")
        summarize_negotiation(context, context.your_offers[-1] if context.your_offers else 0)

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    chat_with_llama()
