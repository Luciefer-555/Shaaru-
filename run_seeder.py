from dotenv import load_dotenv
load_dotenv()
from knowledge_seeder import seed_all
print('Starting knowledge base population...')
result = seed_all()
print('[COMPLETE]', result)
