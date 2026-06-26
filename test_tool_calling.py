from dotenv import load_dotenv
load_dotenv()
from shaaru_brain import _get_client
client = _get_client()
resp = client.chat.completions.create(
    model='meta/llama-3.1-70b-instruct',
    messages=[{'role': 'user', 'content': 'say hi'}],
    tools=[{
        'type': 'function',
        'function': {
            'name': 'test_tool',
            'description': 'a test',
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {'type': 'string'}
                }
            }
        }
    }],
    max_tokens=50
)
print('tool calling supported:', resp)
