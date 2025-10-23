import re
import json
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any
from datetime import datetime
from config import Config
from services.log_utils import Log


class OrderExtractionService:
    """
    Enhanced order extraction using FULL conversation context (Caller + AI).
    
    ✅ KEY IMPROVEMENTS:
    1. Uses BOTH Caller AND AI transcripts for better context
    2. AI confirmations/repetitions validate extracted information
    3. More accurate extraction with conversational understanding
    4. Handles corrections and clarifications naturally
    5. Better extraction for Urdu/Punjabi Roman script
    """
    
    OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
    
    def __init__(self):
        # Store FULL conversation (Caller + AI)
        self._conversation_buffer: List[Dict[str, str]] = []
        
        # Track current order state (ground truth)
        self._current_order: Dict[str, Any] = {
            "customer_name": None,
            "phone_number": None,
            "delivery_address": None,
            "order_items": [],
            "special_instructions": None,
            "payment_method": None,
            "delivery_time": None,
            "total_price": None
        }
        
        self._last_extraction_time: float = 0
        self._extraction_interval: float = 5.0  # Extract every 5 seconds
        self._extraction_task: Optional[asyncio.Task] = None
        self._shutdown: bool = False
        
        # Callback for sending updates
        self.update_callback: Optional[callable] = None
    
    def set_update_callback(self, callback: callable):
        """Set callback for sending order updates to dashboard."""
        self.update_callback = callback
    
    def add_transcript(self, speaker: str, text: str):
        """
        Add transcript from EITHER Caller OR AI.
        
        ✅ Both speakers are tracked for better context!
        """
        if not text or not text.strip():
            return
        
        self._conversation_buffer.append({
            "speaker": speaker,
            "text": text.strip(),
            "timestamp": datetime.now().isoformat()
        })
        
        Log.info(f"[OrderExtraction] Added {speaker}: {text.strip()[:50]}...")
        
        # Keep last 50 messages
        if len(self._conversation_buffer) > 50:
            self._conversation_buffer = self._conversation_buffer[-50:]
        
        # Trigger extraction if enough time has passed
        current_time = asyncio.get_event_loop().time()
        if current_time - self._last_extraction_time >= self._extraction_interval:
            self._trigger_extraction()
    
    def _trigger_extraction(self):
        """Start extraction task."""
        if not self._extraction_task or self._extraction_task.done():
            self._extraction_task = asyncio.create_task(self._extract_order_info())
    
    def _normalize_items(self, items: List[Dict]) -> str:
        """Normalize items list for comparison (ignores order)."""
        if not items:
            return "[]"
        
        sorted_items = sorted(items, key=lambda x: str(x.get("item", "")))
        return json.dumps(sorted_items, sort_keys=True)
    
    def _is_valid_phone(self, phone: str) -> bool:
        """Validate phone number format."""
        if not phone or not isinstance(phone, str):
            return False
        
        cleaned = re.sub(r'[\s\-\(\)]', '', phone)
        digits = re.findall(r'\d', cleaned)
        return len(digits) >= 10
    
    def _is_valid_price(self, price: str) -> bool:
        """Validate price format."""
        if not price or not isinstance(price, str):
            return False
        
        return bool(re.search(r'\d', price))
    
    def get_order_summary(self) -> str:
        """Get human-readable order summary."""
        summary_parts = ["=== ORDER SUMMARY ==="]
        
        if self._current_order.get("customer_name"):
            summary_parts.append(f"Customer: {self._current_order['customer_name']}")
        
        if self._current_order.get("phone_number"):
            summary_parts.append(f"Phone: {self._current_order['phone_number']}")
        
        if self._current_order.get("delivery_address"):
            summary_parts.append(f"Address: {self._current_order['delivery_address']}")
        
        if self._current_order.get("order_items"):
            summary_parts.append("\nItems:")
            for item in self._current_order["order_items"]:
                qty = item.get("quantity", 1)
                name = item.get("item", "Unknown")
                notes = item.get("notes", "")
                item_str = f"  - {qty}x {name}"
                if notes:
                    item_str += f" ({notes})"
                summary_parts.append(item_str)
        
        if self._current_order.get("special_instructions"):
            summary_parts.append(f"\nInstructions: {self._current_order['special_instructions']}")
        
        if self._current_order.get("payment_method"):
            summary_parts.append(f"Payment: {self._current_order['payment_method']}")
        
        if self._current_order.get("delivery_time"):
            summary_parts.append(f"Delivery Time: {self._current_order['delivery_time']}")
        
        if self._current_order.get("total_price"):
            summary_parts.append(f"Total: {self._current_order['total_price']}")
        
        summary_parts.append("=" * 30)
        return "\n".join(summary_parts)
    
    def get_current_order(self) -> Dict[str, Any]:
        """Get current order state."""
        return self._current_order.copy()
    
    async def shutdown(self):
        """Gracefully shutdown."""
        self._shutdown = True
        if self._extraction_task and not self._extraction_task.done():
            self._extraction_task.cancel()
    
    async def _extract_order_info(self):
        """
        Extract structured order information using GPT-4o-mini.
        
        ✅ ENHANCED: Uses FULL conversation including AI responses for better context.
        """
        try:
            self._last_extraction_time = asyncio.get_event_loop().time()
            
            if len(self._conversation_buffer) < 2:
                return
            
            # Build full conversation with BOTH Caller and AI
            conversation_text = "\n".join([
                f"{msg['speaker']}: {msg['text']}" 
                for msg in self._conversation_buffer
            ])
            
            system_prompt = """You are an AI that extracts structured order information from restaurant phone call transcripts.

🔥 CRITICAL RULES:

1. DUAL TRANSCRIPT INPUT:
   - You receive transcripts from BOTH the Caller AND the AI assistant
   - Use AI responses to CONFIRM and VALIDATE customer information
   - AI confirmations are HIGH confidence signals

2. AI CONFIRMATION SIGNALS:
   - When AI says "So you want X", extract X as confirmed
   - When AI says "Your address is Y", extract Y as HIGH confidence
   - When AI says "Your phone number is Z", extract Z as HIGH confidence
   - When AI repeats back information, that's STRONG validation

3. URDU/PUNJABI/ENGLISH MIXED LANGUAGE:
   - Caller transcripts are in Roman/Latin script (e.g., "mera naam Ali hai")
   - Extract names and addresses as written in Roman script
   - Convert to proper capitalization where appropriate
   - Understand phonetic Urdu/Punjabi in English letters

4. EXTRACTION PRIORITY:
   - AI confirmations > Caller statements
   - Repeated information > Single mentions
   - Recent corrections > Old information

5. COMPLETE STATE RESPONSE:
   - Return the COMPLETE current order state (not incremental changes)
   - If customer corrects an item, return the corrected version ONLY
   - If customer removes an item, exclude it from order_items
   - Extract ONLY explicitly confirmed information
   - Return null for anything not clearly mentioned

6. QUALITY FILTERS:
   - Ignore background noise, filler words, and unclear statements
   - If AI asks "Is that correct?" wait for caller's response before extracting
   - Don't extract from questions, only from confirmations

FIELDS TO EXTRACT:
{
  "customer_name": "Full name (Roman script, proper capitalization)",
  "phone_number": "Contact number (validate format: 10+ digits)",
  "delivery_address": "Complete delivery address (Roman script)",
  "order_items": [
    {
      "item": "Item name (capitalize properly)",
      "quantity": number,
      "notes": "Optional special requests"
    }
  ],
  "special_instructions": "Any special delivery/preparation requests",
  "payment_method": "cash/card/online/etc",
  "delivery_time": "When customer wants delivery",
  "total_price": "Total order amount (with currency if mentioned)"
}

CONVERSATION EXAMPLES:

Example 1:
Caller: "mera naam Ali hai"
AI: "Thank you, Ali. What would you like to order?"
→ Extract: {"customer_name": "Ali"}

Example 2:
Caller: "do zinger burger"
AI: "Okay, two zinger burgers. Anything else?"
→ Extract: {"order_items": [{"item": "Zinger Burger", "quantity": 2}]}

Example 3:
Caller: "mera address DHA Phase 5 hai"
AI: "Got it, DHA Phase 5. What's your contact number?"
→ Extract: {"delivery_address": "DHA Phase 5"}

Example 4:
Caller: "mera number 0300-1234567 hai"
AI: "Perfect, 0300-1234567. We'll call you on this number."
→ Extract: {"phone_number": "0300-1234567"} (HIGH confidence - AI confirmed!)

Example 5:
Caller: "nahi wait, teen zinger burger"
AI: "Okay, changing to three zinger burgers."
→ Extract: {"order_items": [{"item": "Zinger Burger", "quantity": 3}]} (Latest correction)

Example 6:
Caller: "I want one large pizza"
AI: "Great! One large pizza. What toppings?"
Caller: "pepperoni aur mushroom"
AI: "Perfect, large pizza with pepperoni and mushroom."
→ Extract: {"order_items": [{"item": "Large Pizza", "quantity": 1, "notes": "pepperoni and mushroom"}]}

RETURN ONLY VALID JSON. No explanations, no markdown, just the JSON object."""

            user_prompt = f"Extract order information from this conversation:\n\n{conversation_text}\n\nReturn JSON only."
            
            headers = {
                "Authorization": f"Bearer {Config.OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 500
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.OPENAI_API_URL,
                    headers=headers,
                    json=payload
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        Log.error(f"[OrderExtraction] API failed: {resp.status} - {error_text}")
                        return

                    try:
                        data = await resp.json()
                    except Exception as e:
                        Log.error(f"[OrderExtraction] JSON decode failed: {e}")
                        return

                    if not data or not isinstance(data, dict):
                        Log.error("[OrderExtraction] Invalid response data type")
                        return

                    choices = data.get("choices")
                    if not choices or not isinstance(choices, list):
                        Log.error("[OrderExtraction] No choices in response")
                        return

                    first_choice = choices[0]
                    if not isinstance(first_choice, dict):
                        Log.error("[OrderExtraction] Invalid choice format")
                        return

                    message = first_choice.get("message")
                    if not message or not isinstance(message, dict):
                        Log.error("[OrderExtraction] No message in choice")
                        return

                    content = message.get("content", "")
                    if not content or not isinstance(content, str):
                        Log.error("[OrderExtraction] Empty or invalid content")
                        return

                    try:
                        # Clean markdown code blocks if present
                        content = content.strip()
                        if content.startswith("```"):
                            content = content.split("```")[1]
                            if content.startswith("json"):
                                content = content[4:]
                        
                        extracted = json.loads(content.strip())

                        # Track what changed
                        updates = {}

                        def update_if_changed(key):
                            """Update field if it changed."""
                            if extracted.get(key) and extracted[key] != self._current_order.get(key):
                                self._current_order[key] = extracted[key]
                                updates[key] = extracted[key]

                        # Basic field updates
                        update_if_changed("customer_name")
                        update_if_changed("delivery_address")
                        update_if_changed("special_instructions")
                        update_if_changed("payment_method")
                        update_if_changed("delivery_time")

                        # Phone validation before accepting
                        if extracted.get("phone_number") and self._is_valid_phone(extracted["phone_number"]):
                            if extracted["phone_number"] != self._current_order.get("phone_number"):
                                self._current_order["phone_number"] = extracted["phone_number"]
                                updates["phone_number"] = extracted["phone_number"]

                        # Validate and update order_items
                        if extracted.get("order_items") and isinstance(extracted["order_items"], list):
                            new_norm = self._normalize_items(extracted["order_items"])
                            old_norm = self._normalize_items(self._current_order.get("order_items", []))
                            if new_norm != old_norm:
                                self._current_order["order_items"] = extracted["order_items"]
                                updates["order_items"] = extracted["order_items"]

                        # Validate price format before updating
                        if extracted.get("total_price") and self._is_valid_price(str(extracted["total_price"])):
                            if extracted["total_price"] != self._current_order.get("total_price"):
                                self._current_order["total_price"] = extracted["total_price"]
                                updates["total_price"] = extracted["total_price"]

                        # Send updates to dashboard
                        if updates and self.update_callback:
                            await self.update_callback(updates)
                            Log.info(f"[OrderExtraction] ✅ Updated: {json.dumps(updates, indent=2)}")

                    except json.JSONDecodeError as e:
                        Log.error(f"[OrderExtraction] JSON parse error: {e}")
                        Log.error(f"[OrderExtraction] Content was: {content[:200]}")

        except Exception as e:
            Log.error(f"[OrderExtraction] Unexpected error: {e}")
            import traceback
            Log.error(f"[OrderExtraction] Traceback: {traceback.format_exc()}")
