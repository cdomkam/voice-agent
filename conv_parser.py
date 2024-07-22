import os
# from openai import OpenAI

# os.environ['OPENAI_API_KEY'] = os.environ['OPEN_API_KEY']
# client = OpenAI()


class ConvParser:
    def __init__(self, message: list[dict]):
        self.message: list[dict] = message
    
    def parse_from_llm(self) -> str:
        prompt_string = f'''
        Separate these into two Python arrays based on the role. If the role type doesn't change in the next line, merge the responses in the new arrays. The indices of both arrays should refer to the same comment and response. Here are the data:
        
        {self.message}
        '''
        
        # response = client.chat.completions.create(
        #             model="gpt-4-turbo",
        #             messages=[
        #                 {"role": "system", "content": "You are a helpful assistant."},
        #                 {"role": "user", "content": prompt_string},
        #             ]
        #             )
        # return response.choices[0].message.content
        NotImplemented
        ...
    
    def parse_from_list(self)->tuple[list,list]:
        assistant_responses = []
        user_comments = []
        current_role = None
        current_content = ""

        for entry in self.message:
            content = entry['content']
            role = entry['role']
            
            if role == current_role:
                # Continue appending content if it's the same role
                current_content += " " + content.strip()
            else:
                # If role changes, save current content to the appropriate array and start new content
                if current_role:
                    if current_role == 'assistant':
                        assistant_responses.append(current_content)
                    elif current_role == 'user':
                        user_comments.append(current_content)
                
                # Reset the current content and update the role
                current_content = content.strip()
                current_role = role

        # Append the last piece of content to the appropriate array
        if current_role == 'assistant':
            assistant_responses.append(current_content)
        elif current_role == 'user':
            user_comments.append(current_content)

        # Ensuring both arrays have the same number of elements
        min_length = min(len(assistant_responses), len(user_comments))
        assistant_responses = assistant_responses[:min_length]
        user_comments = user_comments[:min_length]

        # print("Assistant Responses:", assistant_responses)
        # print("User Comments:", user_comments)
        return assistant_responses, user_comments

def test_conv_parser():
    from testllmmessage import test_message
     
    parser = ConvParser(message=test_message[2:])
    
    comments, responses = parser.parse_from_list()
    
    function_url = 'https://us-central1-gemini-team.cloudfunctions.net/add_conversation_batch'
    # function_url = 'http://localhost:5001/gemini-team/us-central1/add_conversation_batch'
    # function_url = 'https://us-central1-gemini-team.cloudfunctions.net/say_hello'
    data = {
        # "user_id":"c1dbf4a1-b2af-4423-9c07-2d9a98806ff5",
        "user_id":"c66efcb7-1e0a-4d30-a867-cda28e06a845",
        "comments":comments,
        "responses":responses
    }
    
    make_a_request(function_url=function_url, data=data)

if __name__=="__main__":
    from utility import make_a_request
    test_conv_parser()
    