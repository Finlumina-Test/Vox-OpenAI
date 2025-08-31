# Prompting Realtime Models — Field Guide (Paraphrased)

Source and further reading: https://platform.openai.com/docs/guides/realtime-models-prompting

This guide distills the key ideas from the Realtime prompting docs into practical, phone‑ready patterns for this project (Twilio ⇄ OpenAI Realtime). It is written in our own words and tailored to voice use cases.

---

## 1) Core principles for realtime prompting

- Be explicit about role, goals, boundaries, tone, and pacing in the system prompt.
- Keep utterances short. Phone audio is easier to follow with 1–2 short sentences at a time.
- Ask one question per turn. Avoid multi‑part lists unless asked.
- Make interruption safe. If the user speaks, stop immediately and listen.
- Acknowledge and summarize. Confirm key details back to the user.
- Prefer concrete instructions over vague “be helpful”.
- State escalation rules (when to hand off to a human) and what to capture before doing so.

---

## 2) System prompt structure (voice assistants)

Use this structure as a template:

1. Role and objective: who you are and what you must accomplish.
2. Voice and pacing: short sentences, phone‑friendly phrasing, non‑robotic but concise.
3. Safety and policy boundaries: what not to do; how to respond when unsure.
4. Information strategy: what to ask for, one item at a time, and how to confirm.
5. Escalation: how and when to route to a human; which contact details to collect.
6. Tone guardrails: e.g., professional and brand‑safe; no emojis.

See `config.py` → `SYSTEM_MESSAGE` for a ready‑to‑use version (real estate example).

---

## 3) Turn‑taking and interruption handling

Realtime models support conversational turn‑taking. For phone UX:

- Enable voice activity detection (VAD) on input. In this repo, the session uses `server_vad` in `session.update`.
- When you receive `input_audio_buffer.speech_started` from the model, treat it as user interruption.
- On interruption:
	- Truncate the current assistant output with `conversation.item.truncate` using the elapsed audio time.
	- Clear any buffered audio for the caller (Twilio “clear”) and reset local state.
- Keep responses small so interruptions feel natural.

Handled in this repo by `OpenAIService.handle_interruption(...)` and `AudioService` timing/mark logic.

---

## 4) Streaming patterns that feel natural

- Send short, self‑contained sentences to reduce perceived latency.
- Prefer consistent speaking rate and avoid long pauses.
- Avoid reading long unordered lists; offer to send details later or summarize top 2–3 items.
- Use brief confirmations: “Got it”, “Thanks”, followed by the next focused question.

---

## 5) Grounding and domain knowledge

Tell the model what it can and cannot claim:

- Provide domain‑specific constraints (e.g., real estate compliance, fair housing).
- If the model lacks specific listing data, direct it to say it will have an agent follow up.
- Summarize important constraints in the system prompt so they are always active.

---

## 6) Memory, state, and summarization

Realtime conversations can get long. Keep state tidy:

- Summarize periodically: name, contact detail confirmation, preferences, and next action.
- Keep a short rolling memory on your side (server) and include a compact summary as needed in prompts.
- When switching topics, restate the known context in one sentence for stability.

---

## 7) Safety and brand controls

- State prohibited content and compliance rules in the system prompt (e.g., no advice, no discriminatory statements, no steering in real estate).
- Prefer “I can’t speak to that” + a safe alternative or escalation to a human.
- Keep tone consistent with brand voice; avoid jokes unless brand‑approved.

---

## 8) Template snippets (Realtime API)

These JSON snippets match the event types used by this project. Adapt as needed.

### Session configuration

```json
{
	"type": "session.update",
	"session": {
		"type": "realtime",
		"model": "gpt-realtime",
		"output_modalities": ["audio"],
		"audio": {
			"input": {
				"format": {"type": "audio/pcmu"},
				"turn_detection": {"type": "server_vad"}
			},
			"output": {
				"format": {"type": "audio/pcmu"}
			}
		},
		"instructions": "<YOUR SYSTEM PROMPT HERE>"
	}
}
```

### Send caller audio

Append audio chunks (base64) from the phone stream:

```json
{"type": "input_audio_buffer.append", "audio": "<base64-pcmu>"}
```

Then commit the buffer when you want the model to process it:

```json
{"type": "input_audio_buffer.commit"}
```

### Ask the model to speak

```json
{"type": "response.create"}
```

You’ll receive `response.output_audio.delta` frames that you forward to Twilio.

### Handle interruption (user starts speaking)

When you receive `input_audio_buffer.speech_started` from the model:

1) Compute elapsed audio time for the current assistant output.
2) Truncate the in‑flight assistant response:

```json
{
	"type": "conversation.item.truncate",
	"item_id": "<last_assistant_item_id>",
	"content_index": 0,
	"audio_end_ms": 420
}
```

3) Clear any buffered Twilio audio and reset local marks.

---

## 9) Prompt patterns (copy‑adapt)

### A) Phone assistant skeleton

> You are a voice assistant for <COMPANY>. Keep sentences short for phone audio. Ask one question at a time. Stop speaking immediately if the caller talks. Confirm key details briefly. If unsure, say you’ll check and arrange a follow‑up. No emojis or jokes.

Add sections for compliance, disallowed areas, and escalation rules.

### B) Lead qualification checklist (real estate)

- Name and preferred contact (phone or email)
- Buy or rent; property type
- Target neighborhoods
- Budget and timeline
- Beds/baths; must‑haves (parking, pets, yard)
- Tour interest and time window

### C) Safety fallback

> I can’t provide advice on that. If you’d like, I can pass this to a licensed agent to follow up.

---

## 10) Latency tips

- Keep responses short; stream early and often.
- Avoid back‑to‑back multi‑sentence monologues.
- Prefer consistent audio format and sample rate across systems (this repo uses PCMU).

---

## 11) Debugging your prompt

- Turn on timing logs (see `SHOW_TIMING_MATH` in `.env`) during development.
- Log notable events: `speech_started`, `response.*`, buffer commits.
- When behavior drifts, restate constraints at the end of the system prompt.
- Maintain a small “session summary” string that you update every few turns.

---

## 12) End‑to‑end flow in this repo (where things live)

- System prompt: `config.py` → `SYSTEM_MESSAGE` (real estate tuned).
- Session update: `OpenAISessionManager.create_session_update`.
- Send/receive routing: `WebSocketConnectionManager`.
- Audio processing + timing/marks: `AudioService`.
- Interruption handling: `OpenAIService.handle_interruption` + `AudioService` math.

Use this document as a checklist when tuning prompts for new industries.

https://platform.openai.com/docs/guides/realtime-models-prompting
