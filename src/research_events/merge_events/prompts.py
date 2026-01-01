categorize_events_prompt = """
You are a helpful assistant that will categorize the events into the 4 categories.

<Events>
{events}
</Events>

<Categories>
context: Background info, previous relationships, or the 'calm before the storm'. origin of the beef.
conflict: The main incident, the accusation, the leak, the breakup, or the scandal itself.
reaction: Public responses, PR statements, tweets from other influencers, lawsuits, or 'receipts' posted.
outcome: Current status, who was cancelled, impact on career, or final resolution (if any).
</Categories>


<Rules>
INCLUDE ALL THE INFORMATION FROM THE EVENTS, do not abbreviate or omit any information.
</Rules>
"""

EXTRACT_AND_CATEGORIZE_PROMPT = """
You are a Drama/Scandal Event Extractor and Categorizer. Your task is to analyze text chunks for events related to the topic/person.**

<Available Tools>
- `IrrelevantChunk` (use if the text contains NO drama/scandal events relevant to the research question)
- `RelevantEventsCategorized` (use if the text contains relevant events - categorize them into the 4 categories)
</Available Tools>

<Categories>
context: Background info, previous relationships, or the 'calm before the storm'. origin of the beef.
conflict: The main incident, the accusation, the leak, the breakup, or the scandal itself.
reaction: Public responses, PR statements, tweets from other influencers, lawsuits, or 'receipts' posted.
outcome: Current status, who was cancelled, impact on career, or final resolution (if any).
</Categories>

**EXTRACTION RULES**:
- Extract COMPLETE sentences with ALL available details (dates, names, platforms, context, emotions)
- Include "receipts" (screenshots, quotes, specific evidence)
- Preserve the tone of the drama (e.g., if it was a heated argument, describe it as such)
- Include only events directly relevant to the research question
- Maintain chronological order within each category
- Format as clean bullet points with complete, detailed descriptions
- IMPORTANT: Return each category as a SINGLE string containing all bullet points, not as a list

<Text to Analyze>
{text_chunk}
</Text to Analyze>

You must call exactly one of the provided tools. Do not respond with plain text.
"""


MERGE_EVENTS_TEMPLATE = """You are an expert Editor. Merge the 'New Events' into the 'Original Events' list.

<Critical Rules>
1. **DEDUPLICATE**: If an event in 'New Events' is already covered in 'Original Events' (even with slightly different wording), DO NOT add it again. Just merge any new specific details (like a specific time or quote) into the existing bullet point.
2. **FIX BROKEN TEXT**: If a sentence ends abruptly or has escaping characters like "OpenAI\\", fix it or remove the garbage characters.
3. **CHRONOLOGY**: Maintain strict time order.
4. **FORMAT**: Output ONLY clean bullet points.
</Critical Rules>

<Events>
Original events:
{original}

New events:
{new}
</Events>

<Output>
Return only the merged, de-duplicated list of events as bullet points.
</Output>"""
