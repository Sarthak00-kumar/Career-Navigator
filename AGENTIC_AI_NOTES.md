# Career Agent (no LLM)

The app includes a lightweight, rule-based Career Planning Agent. It does not call an LLM or require an API key.

## Agent loop
- Observe: reads the current skill gaps, target role, roadmap, and completed task state.
- Decide: selects the first incomplete task in the current prioritised roadmap.
- Act: presents a concrete learning action and expected deliverable.
- Reassess: Streamlit reruns when a task toggle changes, and the agent selects the next incomplete task.

## Limitations
This is a transparent rule-based agent, not an LLM-powered agent. It does not autonomously browse the web, enrol users in courses, or validate live vacancies. Progress is session-based and should not be described as permanent tracking.
