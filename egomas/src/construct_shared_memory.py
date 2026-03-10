import json
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from google import genai

from egomas.utils.io import load_json, save_json
from egomas.utils.parsing import extract_codeblock_text
from egomas.utils.prompts import SHARED_MEMORY_PROMPT


def _build_tasks(cap_10m_dir, cap_10m_files):
    """Build list of (day, start_time, end_time, context) for all non-empty windows."""
    tasks = []
    for day in range(1, 8):
        day_files = [f for f in cap_10m_files if f"DAY{day}" in f]
        all_day_caps = {}
        for file in day_files:
            data = load_json(os.path.join(cap_10m_dir, file))
            for time, caption in data.items():
                all_day_caps[time] = caption

        start_time = 9000000
        end_time = start_time + 100000
        while start_time < 23000000:
            context = ""
            for time, caption in all_day_caps.items():
                c_start_time = int(time.split("_")[3])
                c_end_time = int(time.split("_")[4])
                if c_start_time >= start_time and c_end_time <= end_time:
                    context += f"{time.split('_')[2].capitalize()}: {caption}\n"

            if context != "":
                tasks.append((day, start_time, end_time, context))

            start_time = end_time
            end_time = start_time + 100000
            if end_time % 1000000 == 600000:
                end_time += 400000
    return tasks


def _call_api_one(args):
    """Worker: call API for one (day, start_time, end_time, context). Must be top-level for pickle."""
    day, start_time, end_time, context, prompt = args
    try:
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key)
        combined = f"{context}\n\n{prompt}"
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=combined,
        )
        content = response.text
        raw = extract_codeblock_text(content)
        memory = json.loads(raw)
        return {"day": day, "start": start_time, "end": end_time, "memory": memory}
    except Exception as e:
        return {"day": day, "start": start_time, "end": end_time, "error": str(e), "memory": content}


def generate_shared_memory_10m(num_workers=None):
    if num_workers is None:
        num_workers = min(cpu_count(), 8)  # cap to avoid API rate limits
    cap_10m_dir = "data/caption/10min"
    cap_10m_files = [f for f in os.listdir(cap_10m_dir) if f.endswith(".json")]

    tasks = _build_tasks(cap_10m_dir, cap_10m_files)
    # Pass prompt in each task so worker process doesn't need egomas on path
    task_args = [(d, s, e, ctx, SHARED_MEMORY_PROMPT) for d, s, e, ctx in tasks]

    memories = []
    with Pool(processes=num_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(_call_api_one, task_args),
            total=len(task_args),
            desc="Shared memory",
        ):
            if result.get("memory") is not None:
                memories.append(
                    {"day": result["day"], "start": result["start"], "end": result["end"], "memory": result["memory"]}
                )
            elif "error" in result:
                print(f"Error (day={result['day']}, start={result['start']}): {result['error']}, content: {result['memory']}")

    memories.sort(key=lambda x: (x["day"], x["start"]))
    save_json(memories, "data/10m_shared_memory.json")


if __name__ == "__main__":
    generate_shared_memory_10m()