from dotenv import load_dotenv
load_dotenv()
from shaaru_brain import _get_client
import json

client = _get_client()

try:
    resp = client.chat.completions.create(
        model='meta/llama-3.1-70b-instruct',
        messages=[{'role': 'user', 'content': 'say hi'}],
        tools=[{
            'type': 'function',
            'function': {
                'name': 'test_tool',
                'description': 'test',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'query': {'type': 'string'}
                    },
                    'required': ['query']
                }
            }
        }],
        tool_choice='auto',
        max_tokens=50,
        timeout=20
    )
    print('SUCCESS')
    print('finish_reason:', resp.choices[0].finish_reason)
    print('tool_calls:', resp.choices[0].message.tool_calls)
    print('content:', resp.choices[0].message.content)
except Exception as e:
    print('ERROR:', type(e).__name__)
    print('DETAIL:', str(e))
