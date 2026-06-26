from dotenv import load_dotenv
load_dotenv()
from shaaru_brain import _get_client

client = _get_client()

try:
    resp = client.chat.completions.create(
        model='meta/llama-3.3-70b-instruct',
        messages=[{'role': 'user', 'content': 'I want this blazer made'}],
        tools=[{
            'type': 'function',
            'function': {
                'name': 'trigger_tailor_flow',
                'description': 'Launch the tailor flow when user wants a garment made',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'garment_description': {'type': 'string'},
                        'confirmed': {'type': 'boolean'}
                    },
                    'required': ['garment_description', 'confirmed']
                }
            }
        }],
        tool_choice='auto',
        max_tokens=100,
        timeout=30
    )
    print('SUCCESS')
    print('finish_reason:', resp.choices[0].finish_reason)
    print('tool_calls:', resp.choices[0].message.tool_calls)
    print('content:', resp.choices[0].message.content)
except Exception as e:
    print('ERROR:', type(e).__name__)
    print('DETAIL:', str(e)[:200])
