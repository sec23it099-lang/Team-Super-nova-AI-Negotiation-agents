# -*- coding: utf-8 -*-
"""
AI NEGOTIATION SELLER - Interactive Chat with LLaMA 3.1:8B
Run: python negotiation_seller_agent.py
"""

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import re
import subprocess
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

# Ollama model
OLLAMA_MODEL = "llama3.1:8b"

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
    seller_minimum_price: int
    current_round: int
    buyer_offers: List[int]
    seller_offers: List[int]
    messages: List[Dict[str, str]]

class DealStatus(Enum):
    ONGOING = "ongoing"
    ACCEPTED = "accepted"
    REJECTED = "rejected"

# ============================================
# BASE AGENT
# ============================================

class BaseSellerAgent(ABC):
    def __init__(self, name: str):
        self.name = name
        self.personality = self.define_personality()

    @abstractmethod
    def define_personality(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def respond_to_buyer_offer(self, context: NegotiationContext, buyer_price: int, buyer_message: str) -> Tuple[DealStatus, int, str]:
        pass

    @abstractmethod
    def get_personality_prompt(self) -> str:
        pass

# ============================================
# SELLER IMPLEMENTATION
# ============================================

class YourSellerAgent(BaseSellerAgent):

    def define_personality(self) -> Dict[str, Any]:
        return {
            "personality_type": "firm but friendly",
            "traits": ["persuasive", "confident", "value-focused", "profit-minded"],
            "catchphrases": [
                "These are top-quality goods.",
                "I think you'll find the price fair for what you get."
            ]
        }

    def respond_to_buyer_offer(
        self,
        context: NegotiationContext,
        buyer_price: int,
        buyer_message: str
    ) -> Tuple[DealStatus, int, str]:

        # ===== Special round-based behavior =====
        if context.current_round == 9:
            counter = int(buyer_price * 1.10)  # 10% increase from buyer's offer
            return DealStatus.ONGOING, counter, f"If you can raise it by 10% to ₹{counter}, we have a deal."

        if context.current_round == 10:
            return DealStatus.ACCEPTED, buyer_price, f"Alright, deal at ₹{buyer_price}!"

        # ===== Normal behavior =====
        fair_price = self.calculate_fair_price(context.product)
        last_offer = context.seller_offers[-1] if context.seller_offers else fair_price + 50

        prompt = f"""
You are the most talented seller, negotiating for {context.product.quantity} x {context.product.name} (quality: {context.product.quality_grade}).
Market price: ₹{context.product.base_market_price}, Fair price: ₹{fair_price}, Minimum acceptable price: ₹{context.seller_minimum_price}.
Buyer offer: ₹{buyer_price} — "{buyer_message}".
Your last offer: ₹{last_offer}.

Rules:
- Never sell at or below market price; always profit.
- Never increase price after lowering it.
- Replies should be persuasive but short (max 2 sentences).
"""

        ai_reply = self.query_ollama(prompt, fallback_price=last_offer)
        counter = self.extract_price(ai_reply)

        # Accept only if buyer meets or exceeds both market price and last offer
        if buyer_price >= context.product.base_market_price and buyer_price >= last_offer:
            return DealStatus.ACCEPTED, buyer_price, f"Deal at ₹{buyer_price}! You’re getting unmatched value."

        # Ensure counteroffer is strictly above market price
        min_price_allowed = context.product.base_market_price + 1
        if counter <= context.product.base_market_price:
            counter = min_price_allowed
            ai_reply = f"Considering the quality of these {context.product.name}, the best I can do is ₹{counter}."

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
            return response if response else f"I can do ₹{fallback_price}."
        except Exception:
            return f"I can do ₹{fallback_price}."

    def extract_price(self, text: str) -> int:
        s = text.replace(",", "").replace("₹", "").strip()
        match = re.search(r"(\d+(\.\d+)?)", s)
        if not match:
            return 0
        return int(float(match.group(1)))

    def calculate_fair_price(self, product: Product) -> int:
        base = product.base_market_price
        grade_adj = {"A": 1.05, "B": 0.95, "Export": 1.10}
        multiplier = grade_adj.get(product.quality_grade, 1.0)
        if product.attributes.get("export_grade"):
            multiplier += 0.02
        fair = int(base * multiplier)
        # Always at least ₹1 above market price
        return max(fair, base + 1)

    def get_personality_prompt(self) -> str:
        return "You are a persuasive seller who always sells above market price."

# ============================================
# INTERACTIVE CHAT
# ============================================

def chat_with_seller_llama():
    print("=== Chat with YourSellerAgent (LLaMA 3.1:8B) ===")
    print("You are the buyer. Type your offer or message. Type 'exit' to quit.")

    agent = YourSellerAgent(name="ChatSeller")

    product = Product(
        name="Alphonso Mangoes",
        category="Fruit",
        origin="India",
        quantity=10,
        base_market_price=600,
        quality_grade="A",
        attributes={"export_grade": True}
    )

    context = NegotiationContext(
        product=product,
        seller_minimum_price=460,
        current_round=0,
        buyer_offers=[],
        seller_offers=[],
        messages=[]
    )

    # AI starts with an opening offer
    opening_price = int(product.base_market_price * 1.15)  # 15% above market
    opening_msg = f"These are premium {product.name}. I can offer them for ₹{opening_price}."
    context.seller_offers.append(opening_price)
    context.messages.append({"role": "seller", "message": opening_msg})
    print(f"AI Seller: ₹{opening_price} — {opening_msg}")

    while True:
        user_input = input("You (Buyer): ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chat...")
            break

        match = re.search(r"(\d+)", user_input.replace(",", ""))
        buyer_price = int(match.group(1)) if match else 0
        buyer_message = user_input

        context.current_round += 1
        context.buyer_offers.append(buyer_price)
        context.messages.append({"role": "buyer", "message": buyer_message})

        status, ai_offer, ai_msg = agent.respond_to_buyer_offer(context, buyer_price, buyer_message)
        context.seller_offers.append(ai_offer)
        context.messages.append({"role": "seller", "message": ai_msg})

        print(f"AI Seller: ₹{ai_offer} — {ai_msg}")

        if status == DealStatus.ACCEPTED:
            print(f"--- DEAL MADE at ₹{ai_offer}! ---")
            break

if __name__ == "__main__":
    chat_with_seller_llama()
