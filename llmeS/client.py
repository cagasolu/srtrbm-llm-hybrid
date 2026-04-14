# GLOBAL LLM CLIENT (FULLY AGNOSTIC + SAFE)
#
# - Works with ANY backend
# - Never crashes the system
# - Normalizes output

import os
import time


class BaseLLM:
    """
    User must implement this.

    Expected:
        generate(prompt, images=None) -> str | dict | None
    """

    def __init__(self, api_key=None):
        self.api_key = api_key

    def generate(self, prompt, images=None):
        raise NotImplementedError


class SafeBookClient:

    def __init__(self):

        api_key = os.getenv("LLM_API_KEY")

        self.enabled = True

        try:
            self.backend = BaseLLM(api_key=api_key)

        except Exception as e:
            print("[LLM DISABLED]", str(e))
            self.backend = None
            self.enabled = False

    def _normalize(self, result):
        """
        Normalize ANY backend output → string
        """

        if result is None:
            return None

        # string
        if isinstance(result, str):
            return result

        # dict with text
        if isinstance(result, dict):
            if "text" in result:
                return result["text"]

            if "output" in result:
                return result["output"]

        # fallback
        try:
            return str(result)
        except Exception:
            return None

    def generate(self, prompt, images=None):

        if not self.enabled or self.backend is None:
            return None

        max_retries = 3
        delay = 1.5

        for attempt in range(max_retries):
            try:
                result = self.backend.generate(prompt, images=images)
                return self._normalize(result)

            except Exception as e:
                if attempt == max_retries - 1:
                    print("[LLM ERROR]", str(e))
                    return None

                time.sleep(delay * (attempt + 1))

        return None


# GLOBAL INSTANCE
client = SafeBookClient()