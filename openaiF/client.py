from openai import OpenAI
import os
import time


class SafeOpenAIClient:

    def __init__(self):
        """
        HOW TO SET API KEY:

        Option 1 (Manual):
            api_key = "sk-xxxx"

        Option 2 (Environment):
            export OPENAI_API_KEY="your_key_here"
        """

        # manual
        api_key = ""

        # fallback
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")

        self.enabled = True

        if not api_key:
            print("[LLM DISABLED] No API key provided")
            self.enabled = False
            self.client = None
        else:
            self.client = OpenAI(api_key=api_key)

    def create_response(self, **kwargs):
        if not self.enabled:
            return None

        max_retries = 3
        delay = 1.5

        for attempt in range(max_retries):
            try:
                return self.client.responses.create(**kwargs)

            except Exception as era:
                if attempt == max_retries - 1:
                    print("[OpenAI ERROR]", str(era))
                    return None

                time.sleep(delay * (attempt + 1))

        return None


# Global instance
client = SafeOpenAIClient()