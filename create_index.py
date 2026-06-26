from dotenv import load_dotenv
load_dotenv()
from shaaru_brain import _get_db
db = _get_db()
db['products'].create_index([('name', 'text'), ('brand', 'text')])
