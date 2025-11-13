import os
import time
import json
from openai import OpenAI

_verbose = False

class OpenAIClient():
    model_path = "gpt:"
    
    _params = {
        "temperature": 0.0,
        "top_p": 1,
        "max_tokens": 400,
        "seed": 100,
    }

    def get_params(self):
        return dict(self._params)

    def __init__(self, model_id, params=None):

        OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

        if OPENAI_API_KEY is None or OPENAI_API_KEY == "":
            raise Exception(f"To use OpenAI models, you need to set up the env var OPENAI_API_KEY.")
        
        self.client = OpenAI(
            api_key=OPENAI_API_KEY
        )

        self.model_id = model_id
        self.params = dict(self._params)
        self.params['model'] = self.model_id
        if params is not None:
            self.params.update(params)

    def generate_response(self, user_input: str, exit_on_exception=False) -> str:
        
        system_input = "You are a helpful assistant."
        
        data = dict(self.params)
        data.update({
            "messages": [
                {
                    "role": "system",
                    "content": system_input,
                },
                {
                    "role": "user",
                    "content": user_input,
                }
            ],
        })
        
        if _verbose:
            print(json.dumps(data, indent=4))

        try:
            time0 = time.time()
            response = self.client.chat.completions.create(**data)
            time1 = time.time()
            elapsed = time1 - time0
            if _verbose:
                print(f"Used {elapsed} seconds.")
        except Exception as e:
            print(f"Failed to call OpenAI {self.model_id}: {str(e)}")
            raise

        if _verbose:
            print(response)

        try:
            output = response.choices[0].message.content.strip()
        except Exception as e:
            if _verbose:
                print("Exception calling OpenAI", response)
                print(json.dumps(response.model_dump(), indent=2))
            output = None 
            
        return output


if __name__ == "__main__":
    
    model_id = 'gpt-4o-mini'
    client = OpenAIClient(model_id)

    user_input = "What is 2+3?"

    try:
        response = client.generate_response(user_input)
        print("Response from OpenAI:", response)
    except Exception as e:
        print(f"An error occurred: {e}")