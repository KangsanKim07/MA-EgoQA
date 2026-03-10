SHARED_MEMORY_PROMPT = """These are captions of 10 minutes video captured from multiple people. Write a structured summary of the captions.
The output should consist of multiple items in the following format:
{
    "name": {
        "type": List[str],
        "description": "All names that involved in the event.",
    },
    "action": {
        "type": str,
        "description": "Action that happened in the event.",
    },
    "location": {
        "type": str,
        "description": "Location of the action.",
    },
    "detail": {
        "type": str,
        "description": "Important details of the action.",
    },
}

For example, if there are 3 key events in the context, the output should be like this:
[
    {
        "name": ["Jake", "Katrina"],
        "action": "Prepare for the lunch. Make pancakes and omelettes.",
        "location": "Kitchen",
        "detail": "Jake first made batter for the pancakes using powder and water, saying 'I need to make batter for the pancakes.'. And then he turned on the stove ...",
    },
    {
        "name": ["Lucia", "Jake", "Katrina", "Tasha"],
        "action": "Have a lunch together. Discuss about the plan for the party.",
        "location": "Living room",
        "detail": "Lucia brought the plates and placed them on the table, saying 'Let’s get started with lunch.'. Jake served the pancakes and omelettes to everyone. Katrina poured some juice into glasses. Tasha sat down and thanked Jake for cooking. As they ate, Lucia asked, 'So, what’s the plan for the party?'. ..."
    },
    {
        "name": ["Shure"],
        "action": "Play the piano.",
        "location": "Living room",
        "detail": "Shure sat down at the piano and started playing. He played a few notes and then started playing a song. He played the song for a while and then ..."
    }
]

Try to contain as many key events as possible. Do not miss any important events from the context. Also, put all the details of the action in the detail field.
"""

# ---------------------------------------------------------------------------
# Planner prompt (name/query selection)
# ---------------------------------------------------------------------------
PLANNER_SYSTEM = """You are helpful planner choosing which person's perspective should the response model refer to during answering the question.

### Context
{context}

### Question
{question}

Do not answer the question now. Follow the instructions below.

Based on the context, select one person (should be one of the list: Jake, Alice, Katrina, Lucia, Tasha, Shure) and generate query which memory should be referred to from that person. The question could be a sub-question for the original question.

For example, the answer should be like this: [{{'name' : 'Jake', 'query' : 'order McDonalds for lunch'}}, {{'name' : 'Katrina', 'query' : 'made pancakes'}}, {{'name' : 'Lucia', 'query' : 'clean the kitchen'}}]

Try to generate multiple elements in the list if necessary. Make sure that the output is a list of dictionaries."""

# ---------------------------------------------------------------------------
# Answer prompt (final A–E prediction)
# ---------------------------------------------------------------------------
ANSWER_HEADER = """Answer only the header of the correct option letter (A, B, C, D, E) in the question.

### Shared Context
{shared_context}

### Retrieved Contexts
{retrieved_contexts}

### Question
{question}

Answer only the header of the correct answer. Do not provide any other reasoning. Your answer should be a single letter (A, B, C, D, E)."""