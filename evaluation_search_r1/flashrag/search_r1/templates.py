SEARCH_R1_TEMPLATE = (
"Answer the given question. "
"You must conduct reasoning inside <think> and </think> first every time you get new information. "
"After reasoning, if you find you lack some knowledge, you can call a search engine by "
"<search> query </search> and it will return the top searched results between "
"<information> and </information>. "
"You can search as many times as your want. "
"If you find no further external knowledge needed, you can directly provide the answer inside "
"<answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. "
"Question: {prompt}\n"
)


SEARCH_R1_TEMPLATE_SYS = """You are a helpful assistant that solves the question step by step with a search tool.

Use this protocol exactly:
1) Think with <think>...</think>
2) Search with <search>query</search> when needed
3) Consume evidence from <information>...</information>
4) End with <answer>final answer</answer>

Do not output tool JSON, and do not use <tool_call> or <tool_response> tags."""

