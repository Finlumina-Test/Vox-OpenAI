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
    Only sends confirmed information (no nulls/empty values).
    """
    
    OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
    
    def __init__(self):
        self._conversation_buffer: List[Dict[str, str]] = []
        
        # Track what's been sent to avoid duplicates
        self._sent_data: Dict[str, Any] = {
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
    
    async def _extract_order_info(self):
        """Extract structured order information using GPT."""
        try:
            self._last_extraction_time = asyncio.get_event_loop().time()
            
            if len(self._conversation_buffer) < 2:
                return
            
            conversation_text = "\n".join([
                f"{msg['speaker']}: {msg['text']}" 
                for msg in self._conversation_buffer
            ])
            
            system_prompt = """You are an AI that extracts structured order information from restaurant phone call transcripts.

Extract ONLY confirmed information. Return null for anything not clearly mentioned.

IMPORTANT: Always return the COMPLETE current order state. If items are corrected or changed, return the corrected version, not additions.

Fields to extract:
- customer_name: Full name of customer
- phone_number: Phone number (any format)
- delivery_address: Complete delivery address
- order_items: Array of {"item": "name", "quantity": number, "notes": "optional"} - COMPLETE list of current items
- special_instructions: Any special requests
- payment_method: "cash", "card", or "online"
- delivery_time: Preferred time or "ASAP"
- total_price: Total amount with currency

Return ONLY valid JSON. Example:
{
  "customer_name": "Ahmed Khan",
  "phone_number": "0300-1234567",
  "delivery_address": "House 123, Street 5, DHA Phase 2",
  "order_items": [
    {"item": "Chicken Biryani", "quantity": 2, "notes": "extra spicy"}
  ],
  "special_instructions": "Call before delivery",
  "payment_method": "cash",
  "delivery_time": "7:30 PM",
  "total_price": "1200 PKR"
}"""
            
            user_prompt = f"""Extract order information:\n\n{conversation_text}\n\nReturn JSON only."""
            
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
                    
                    # ✅ FIX: Robust null checking for order extraction
                    try:
                        data = await resp.json()
                    except Exception as e:
                        Log.error(f"[OrderExtraction] JSON decode failed: {e}")
                        return
                    
                    if not data or not isinstance(data, dict):
                        Log.error(f"[OrderExtraction] Invalid response data type")
                        return
                    
                    choices = data.get("choices")
                    if not choices or not isinstance(choices, list) or len(choices) == 0:
                        Log.error(f"[OrderExtraction] No choices in response")
                        return
                    
                    first_choice = choices[0]
                    if not isinstance(first_choice, dict):
                        Log.error(f"[OrderExtraction] Invalid choice format")
                        return
                    
                    message = first_choice.get("message")
                    if not message or not isinstance(message, dict):
                        Log.error(f"[OrderExtraction] No message in choice")
                        return
                    
                    content = message.get("content", "")
                    if not content or not isinstance(content, str):
                        Log.error(f"[OrderExtraction] Empty or invalid content")
                        return
                    
                    try:
                        content = content.strip()
                        if content.startswith("```"):
                            content = content.split("```")[1]
                            if content.startswith("json"):
                                content = content[4:]
                        
                        extracted = json.loads(content.strip())
                        
                        # Send only NEW or CHANGED data
                        updates = {}
                        
                        # Check each field for NEW or CHANGED data
                        if extracted.get("customer_name") and extracted["customer_name"] != self._sent_data["customer_name"]:
                            updates["customer_name"] = extracted["customer_name"]
                            self._sent_data["customer_name"] = extracted["customer_name"]
                        
                        if extracted.get("phone_number") and extracted["phone_number"] != self._sent_data["phone_number"]:
                            updates["phone_number"] = extracted["phone_number"]
                            self._sent_data["phone_number"] = extracted["phone_number"]
                        
                        if extracted.get("delivery_address") and extracted["delivery_address"] != self._sent_data["delivery_address"]:
                            updates["delivery_address"] = extracted["delivery_address"]
                            self._sent_data["delivery_address"] = extracted["delivery_address"]
                        
                        if extracted.get("special_instructions") and extracted["special_instructions"] != self._sent_data["special_instructions"]:
                            updates["special_instructions"] = extracted["special_instructions"]
                            self._sent_data["special_instructions"] = extracted["special_instructions"]
                        
                        if extracted.get("payment_method") and extracted["payment_method"] != self._sent_data["payment_method"]:
                            updates["payment_method"] = extracted["payment_method"]
                            self._sent_data["payment_method"] = extracted["payment_method"]
                        
                        if extracted.get("delivery_time") and extracted["delivery_time"] != self._sent_data["delivery_time"]:
                            updates["delivery_time"] = extracted["delivery_time"]
                            self._sent_data["delivery_time"] = extracted["delivery_time"]
                        
                        if extracted.get("total_price") and extracted["total_price"] != self._sent_data["total_price"]:
                            updates["total_price"] = extracted["total_price"]
                            self._sent_data["total_price"] = extracted["total_price"]
                        
                        # Handle order items - REPLACE entire list if changed (allows corrections)
                        if extracted.get("order_items") and isinstance(extracted["order_items"], list):
                            # Normalize both lists for comparison (convert to JSON strings)
                            new_items_normalized = json.dumps(extracted["order_items"], sort_keys=True)
                            old_items_normalized = json.dumps(self._sent_data["order_items"], sort_keys=True)
                            
                            # If items changed at all, replace the entire list
                            if new_items_normalized != old_items_normalized:
                                self._sent_data["order_items"] = extracted["order_items"]
                                updates["order_items"] = extracted["order_items"]
                        
                        # Send updates if we have new data
                        if updates and self.update_callback:
                            await self.update_callback(updates)
                            Log.info(f"[OrderExtraction] Updated data: {json.dumps(updates, indent=2)}")
                        
                    except json.JSONDecodeError as e:
                        Log.error(f"[OrderExtraction] JSON parse error: {e}")
            
        except Exception as e:
            Log.error(f"[OrderExtraction] Unexpected error: {e}")
    
    def get_current_order(self) -> Dict[str, Any]:
        """Get all extracted order data."""
        return self._sent_data.copy()
    
    def get_order_summary(self) -> str:
        """Get human-readable summary."""
        data = self._sent_data
        lines = ["📋 ORDER SUMMARY", "=" * 50]
        
        if data["customer_name"]:
            lines.append(f"👤 Customer: {data['customer_name']}")
        
        if data["phone_number"]:
            lines.append(f"📞 Phone: {data['phone_number']}")
        
        if data["delivery_address"]:
            lines.append(f"📍 Address: {data['delivery_address']}")
        
        if data["order_items"]:
            lines.append("\n🍽️ ORDER ITEMS:")
            for item in data["order_items"]:
                qty = item.get("quantity", 1)
                name = item.get("item", "Unknown")
                notes = item.get("notes")
                line = f"  • {qty}x {name}"
                if notes:
                    line += f" ({notes})"
                lines.append(line)
        
        if data["special_instructions"]:
            lines.append(f"\n📝 Instructions: {data['special_instructions']}")
        
        if data["payment_method"]:
            lines.append(f"💳 Payment: {data['payment_method']}")
        
        if data["delivery_time"]:
            lines.append(f"⏰ Delivery: {data['delivery_time']}")
        
        if data["total_price"]:
            lines.append(f"💰 Total: {data['total_price']}")
        
        lines.append("=" * 50)
        
        return "\n".join(lines)
    
    def is_order_complete(self) -> bool:
        """
        Check if all essential order information is captured.
        Essential fields: customer_name, phone_number, delivery_address, order_items
        """
        essential_fields = ["customer_name", "phone_number", "delivery_address", "order_items"]
        return all(
            self._sent_data.get(field) and 
            (not isinstance(self._sent_data[field], list) or len(self._sent_data[field]) > 0)
            for field in essential_fields
        )
    
    def reset(self):
        """Reset order data for new call."""
        self._conversation_buffer.clear()
        self._sent_data = {
            "customer_name": None,
            "phone_number": None,
            "delivery_address": None,
            "order_items": [],
            "special_instructions": None,
            "payment_method": None,
            "delivery_time": None,
            "total_price": None
        }
    
    async def shutdown(self):
        """Graceful shutdown."""
        self._shutdown = True
        if self._extraction_task and not self._extraction_task.done():
            self._extraction_task.cancel()
            try:
                await self._extraction_task
            except asyncio.CancelledError:
                pass
