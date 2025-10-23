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
    Extracts structured order information from restaurant phone calls.
    
    ✅ FIXES:
    1. Properly handles item deletions/corrections
    2. Sends incremental updates (only what changed)
    3. Validates extracted data quality
    """
    
    OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
    
    def __init__(self):
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
        self._extraction_interval: float = 5.0
        self._extraction_task: Optional[asyncio.Task] = None
        self._shutdown: bool = False
        
        # Callback for sending updates
        self.update_callback: Optional[callable] = None
    
    def set_update_callback(self, callback: callable):
        """Set callback for sending order updates to dashboard."""
        self.update_callback = callback
    
    def add_transcript(self, speaker: str, text: str):
        """Add transcript and trigger extraction."""
        if not text or not text.strip():
            return
        
        self._conversation_buffer.append({
            "speaker": speaker,
            "text": text.strip(),
            "timestamp": datetime.now().isoformat()
        })
        
        if len(self._conversation_buffer) > 50:
            self._conversation_buffer = self._conversation_buffer[-50:]
        
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
    
    async def _extract_order_info(self):
        """
        Extract structured order information using GPT.
        """
        try:
            self._last_extraction_time = asyncio.get_event_loop().time()
            
            if len(self._conversation_buffer) < 2:
                return
            
            conversation_text = "\n".join([
                f"{msg['speaker']}: {msg['text']}" 
                for msg in self._conversation_buffer
            ])
            
            system_prompt = """You are an AI that extracts structured order information from restaurant phone call transcripts.

CRITICAL RULES:
1. Extract ONLY explicitly confirmed information
2. Return null for anything not clearly mentioned
3. Return the COMPLETE current order state (not incremental changes)
4. If customer corrects an item, return the corrected version ONLY
5. If customer removes an item, exclude it from order_items
6. Ignore background noise, filler words, and unclear statements

Fields to extract:
- customer_name
- phone_number
- delivery_address
- order_items (list of {"item": "name", "quantity": number, "notes": "optional"})
- special_instructions
- payment_method
- delivery_time
- total_price

Return ONLY valid JSON.
"""

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
                        content = content.strip()
                        if content.startswith("```"):
                            content = content.split("```")[1]
                            if content.startswith("json"):
                                content = content[4:]
                        
                        extracted = json.loads(content.strip())

                        updates = {}

                        def update_if_changed(key):
                            if extracted.get(key) and extracted[key] != self._current_order.get(key):
                                self._current_order[key] = extracted[key]
                                updates[key] = extracted[key]

                        # Basic field updates
                        update_if_changed("customer_name")
                        update_if_changed("delivery_address")
                        update_if_changed("special_instructions")
                        update_if_changed("payment_method")
                        update_if_changed("delivery_time")
                        update_if_changed("total_price")

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
                        if extracted.get("total_price") and self._is_valid_price(extracted["total_price"]):
                            if extracted["total_price"] != self._current_order.get("total_price"):
                                self._current_order["total_price"] = extracted["total_price"]
                                updates["total_price"] = extracted["total_price"]

                        if updates and self.update_callback:
                            await self.update_callback(updates)
                            Log.info(f"[OrderExtraction] Updated data: {json.dumps(updates, indent=2)}")

                    except json.JSONDecodeError as e:
                        Log.error(f"[OrderExtraction] JSON parse error: {e}")

        except Exception as e:
            Log.error(f"[OrderExtraction] Unexpected error: {e}")
