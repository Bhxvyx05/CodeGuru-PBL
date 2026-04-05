import os
from dotenv import load_dotenv

load_dotenv()
print(os.getenv("GOOGLE_API_KEY")[:6])  # should print 'AIzaSy'
